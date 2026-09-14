"""
BIDS dataset writer.

Exports Session and Study contents as a BIDS directory tree. The reader
half of the round trip is :meth:`~milob.core.study.Study.load_bids_data`,
and the two share their filename conventions.

The export is driven by Session objects rather than filenames: each
session supplies its identifiers and metadata, and each stream its
acquisition metadata, probe geometry and events. A stream registered as a
path has its file copied; one held in memory is written out.

Only continuous-wave recordings can be exported, since the BIDS
channel-type vocabulary is entirely continuous-wave.
"""

import json
import os
import re
import shutil
import warnings

import numpy as np
import pandas as pd

from .snirf import describe_measurement_columns, _get_datatype_names


# ---------------------------------------------------------------------------
# Channel typing
# ---------------------------------------------------------------------------

#: SNIRF ``dataType`` -> (BIDS ``channels.tsv`` type, fallback ``units``).
#: Only raw, non-label-carrying data types appear here; processed data
#: (``dataType`` 99999) is keyed on its ``dataTypeLabel`` instead, below.
#:
#: This table is the whole of BIDS 1.10's NIRS channel-type vocabulary that
#: MILOB can currently produce. The specification's full enum is
#: NIRSCWAMPLITUDE, NIRSCWFLUORESCENSEAMPLITUDE, NIRSCWOPTICALDENSITY,
#: NIRSCWHBO, NIRSCWHBR and NIRSCWMUA -- every one of them continuous-wave.
#: (The spelling 'FLUORESCENSE' is the specification's own; it is not a typo
#: here.)
_BIDS_CHANNEL_TYPES = {
    1:  ('NIRSCWAMPLITUDE', 'unitless'),
    51: ('NIRSCWFLUORESCENSEAMPLITUDE', 'unitless'),
}

#: SNIRF ``dataTypeLabel`` -> (BIDS type, fallback ``units``), for processed
#: columns (``dataType`` 99999). Labels are those emitted by
#: ``io.snirf._enumerate_columns()``.
_BIDS_PROCESSED_TYPES = {
    'dOD': ('NIRSCWOPTICALDENSITY', 'unitless'),
    'HbO': ('NIRSCWHBO', 'uM'),
    'HbR': ('NIRSCWHBR', 'uM'),
    'mua': ('NIRSCWMUA', '1/cm'),
}


#: Channel types for the modalities BIDS does not yet cover, used only by
#: ``mode='extended'``. These follow the
#: specification's own NIRSCW* naming, extended to the SNIRF dataType ranges
#: it has no vocabulary for -- they are milob's proposal, not BIDS: a dataset
#: written with them will fail validation on the channels.tsv type enum.
_PROPOSED_CHANNEL_TYPES = {
    101: ('NIRSFDACAMPLITUDE', 'unitless'),
    102: ('NIRSFDPHASE', 'rad'),
    151: ('NIRSFDFLUORESCENCEAMPLITUDE', 'unitless'),
    152: ('NIRSFDFLUORESCENCEPHASE', 'rad'),
    201: ('NIRSTDGATEDAMPLITUDE', 'unitless'),
    251: ('NIRSTDGATEDFLUORESCENCEAMPLITUDE', 'unitless'),
    301: ('NIRSTDMOMENTAMPLITUDE', 'unitless'),
    351: ('NIRSTDMOMENTFLUORESCENCEAMPLITUDE', 'unitless'),
    401: ('NIRSDCSG2', 'unitless'),
    410: ('NIRSDCSBFI', 'cm2/s'),
}

#: How to treat a stream whose modality BIDS has no channel type for. See
#: ``write_bids_dataset``'s docstring for the trade-off between them.
MODES = ('strict', 'compat', 'extended')


def _unsupported_channel_message(column, stream):
    """Explain why a measurement column has no BIDS channel type."""
    datatype_name = _get_datatype_names([column.datatype])[0]

    if column.label is not None:
        what = f"processed data labelled {column.label!r}"
    else:
        what = f"SNIRF dataType {column.datatype} ({datatype_name})"

    return (
        f"{type(stream).__name__} {stream.name!r} contains {what}, which has "
        "no channel type in BIDS 1.10: the specification's NIRS "
        "'type' vocabulary defines six values and all six are "
        "continuous-wave (NIRSCWAMPLITUDE, NIRSCWFLUORESCENSEAMPLITUDE, "
        "NIRSCWOPTICALDENSITY, NIRSCWHBO, NIRSCWHBR, NIRSCWMUA). Time-domain, "
        "frequency-domain and DCS recordings are valid SNIRF -- "
        "NirsStream.to_snirf() writes them -- but cannot yet be described by "
        "a valid channels.tsv under mode='strict'.\n"
        "Two ways forward, both of which still write the .snirf itself, which "
        "describes the data completely:\n"
        "  mode='compat'   -- type these channels MISC, with the real "
        "modality in the description column. The dataset validates; a reader "
        "recovers the modality from the description or the SNIRF.\n"
        "  mode='extended' -- write proposed type names "
        f"({_PROPOSED_CHANNEL_TYPES.get(column.datatype, ('...',))[0]}, ...). "
        "Semantically complete, but the dataset will NOT validate until BIDS "
        "adopts them."
    )


def _bids_channel_type(column, stream, mode='strict'):
    """
    Map a measurement column to its BIDS channel type and units.

    Units are resolved in decreasing order of authority: the column's own
    recorded unit, then the stream's unit attribute, then a fallback for that
    channel type.

    Parameters
    ----------
    column : MeasurementColumn
        Column to classify.
    stream : NirsStream
        Stream the column belongs to.
    mode : {'strict', 'compat', 'extended'}
        How to treat a modality BIDS has no type for.

    Returns
    -------
    tuple of (str, str, str or None)
        The channel type, its units, and a description note, the last set only
        in compatibility mode, where the type written is MISC and the real
        modality has to be recorded somewhere.
    """
    if column.datatype == 99999:
        entry = _BIDS_PROCESSED_TYPES.get(column.label)
    else:
        entry = _BIDS_CHANNEL_TYPES.get(column.datatype)

    note = None

    if entry is None and mode == 'extended':
        entry = _PROPOSED_CHANNEL_TYPES.get(column.datatype)

    if entry is None and mode == 'compat':
        # MISC is a real value in the enum, so the dataset validates. The
        # modality it stands for is not recoverable from the type alone, so
        # it goes in the optional description column -- lossy, but stated
        # rather than silently dropped, and the .snirf beside it is exact.
        datatype_name = _get_datatype_names([column.datatype])[0]
        entry = ('MISC', _PROPOSED_CHANNEL_TYPES.get(
            column.datatype, (None, 'unitless'))[1])
        note = (f"{datatype_name} (SNIRF dataType {column.datatype}); "
                "BIDS has no channel type for this modality")

    if entry is None:
        raise ValueError(_unsupported_channel_message(column, stream))

    channel_type, fallback_units = entry
    units = column.unit or stream.data.attrs.get('units') or fallback_units

    return channel_type, units, note


# ---------------------------------------------------------------------------
# Filename entities
# ---------------------------------------------------------------------------

#: Inverse of ``Study._extract_bids_stream_name()``: a stream key is
#: ``task[_recording-<label>][_run-<index>]``.
_STREAM_NAME_RE = re.compile(
    r'^(?P<task>.+?)'
    r'(?:_recording-(?P<recording>[^_]+))?'
    r'(?:_run-(?P<run>[0-9]+))?$'
)

#: BIDS labels are alphanumeric; indices are digits.
_LABEL_RE = re.compile(r'^[a-zA-Z0-9]+$')


def entities_from_stream_name(stream_name):
    """
    Recover BIDS filename entities from a stream key.

    The inverse of the key construction used when reading a dataset, so that a
    read followed by a write reproduces the original filenames.

    Parameters
    ----------
    stream_name : str
        Stream key, e.g. 'nback' or 'touch_recording-infant_run-2'.

    Returns
    -------
    dict
        Key 'task', plus 'recording' and 'run' where present.
    """
    match = _STREAM_NAME_RE.match(stream_name)
    if match is None:                       # pragma: no cover - regex is total
        return {'task': stream_name}

    entities = {'task': match.group('task')}

    if match.group('recording'):
        entities['recording'] = match.group('recording')
    if match.group('run'):
        entities['run'] = match.group('run')

    return entities


#: Runs of anything a BIDS label cannot contain.
_NON_LABEL_RE = re.compile(r'[^a-zA-Z0-9]+')


def sanitise_label(value, entity='label'):
    """
    Convert a name into a BIDS filename label.

    BIDS labels are alphanumeric, since underscores and hyphens separate
    entities in a filename, while stream names are ordinary Python
    identifiers. Names are converted to camelCase rather than rejected.

    Parameters
    ----------
    value : str
        Name to convert.
    entity : str
        Entity the label belongs to, used in the error message.

    Returns
    -------
    str
        The sanitised label.

    Raises
    ------
    ValueError
        If nothing alphanumeric remains.

    Examples
    --------
    >>> sanitise_label('cw_stream', 'task')
    'cwStream'
    >>> sanitise_label('finger tapping (left)', 'task')
    'fingerTappingLeft'
    """
    pieces = [piece for piece in _NON_LABEL_RE.split(str(value)) if piece]

    if not pieces:
        raise ValueError(
            f"Cannot build a BIDS {entity} label from {value!r}: it has no "
            "alphanumeric characters, and BIDS labels are alphanumeric "
            "('_' and '-' separate entities in a filename)."
        )

    return pieces[0] + "".join(piece[:1].upper() + piece[1:]
                               for piece in pieces[1:])


def _sanitise_entities(subject, session, entities, renames):
    """
    Sanitise every label of one recording.

    Parameters
    ----------
    subject : str
        Subject identifier.
    session : str
        Session identifier.
    entities : dict
        Further filename entities.
    renames : dict
        Accumulates each original-to-sanitised mapping across the export, so a
        rename can be reported once rather than per subject.

    Returns
    -------
    tuple of (str, str, dict)
        The sanitised subject, session and entities.
    """
    def convert(value, entity):
        if value is None:
            return None
        original = str(value)
        label = sanitise_label(original, entity)
        if label != original:
            renames[(entity, original)] = label
        return label

    return (
        convert(subject, 'subject'),
        convert(session, 'session'),
        {key: convert(value, key) for key, value in (entities or {}).items()},
    )


def _build_prefix(subject, session, entities=None):
    """
    Assemble a BIDS filename stem in the specification's entity order.

    Labels are expected to be sanitised already.

    Parameters
    ----------
    subject : str
        Subject identifier.
    session : str
        Session identifier.
    entities : dict
        Further filename entities.

    Returns
    -------
    str
        The filename stem.
    """
    parts = [f"sub-{subject}"]

    if session is not None:
        parts.append(f"ses-{session}")

    if entities:
        for entity in ('task', 'recording', 'run'):
            if entity in entities:
                parts.append(f"{entity}-{entities[entity]}")

    return "_".join(parts)


# ---------------------------------------------------------------------------
# Sidecar writers
# ---------------------------------------------------------------------------

def _write_json(payload, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)


#: nirs.json fields a caller may supply. All are RECOMMENDED by BIDS and none
#: can be read off a recording -- they describe the instrument, the cap, the
#: institution and the task protocol.
_NIRS_FIELDS = (
    'Manufacturer', 'ManufacturersModelName', 'SoftwareVersions',
    'DeviceSerialNumber', 'CapManufacturer', 'CapManufacturersModelName',
    'SourceType', 'DetectorType', 'HeadCircumference', 'HardwareFilters',
    'SubjectArtefactDescription', 'TaskDescription', 'Instructions',
    'CogAtlasID', 'CogPOID', 'InstitutionName', 'InstitutionAddress',
    'InstitutionalDepartmentName', 'NIRSPlacementScheme',
)

#: SNIRF metaDataTag -> BIDS nirs.json field, for the instrument tags
#: ``read_snirf()`` preserves. Acquisition software commonly writes these, so
#: a real instrument file fills them in without the caller doing anything.
_SNIRF_TO_BIDS = {
    'ManufacturerName': 'Manufacturer',
    'Model': 'ManufacturersModelName',
}


def _derived_nirs_metadata(stream):
    """
    Collect the recommended sidecar fields the recording can answer itself.

    Reads the instrument tags carried by the source file, the recording
    duration, and the short-channel count. A caller's own value always wins.

    Parameters
    ----------
    stream : NirsStream
        Stream to read from.

    Returns
    -------
    dict
        Sidecar fields.
    """
    derived = {}

    for snirf_tag, bids_field in _SNIRF_TO_BIDS.items():
        value = stream.data.attrs.get(snirf_tag)
        if value:
            derived[bids_field] = str(value)

    time = stream.data.coords.get('time')
    if time is not None and time.size > 1:
        derived['RecordingDuration'] = float(
            time.values[-1] - time.values[0]
        )

    return derived


def _channel_counts(channel_table):
    """
    Count the channels BIDS defines against the channel table.

    Both counts are row counts of that file, so they exceed the corresponding
    counts on a probe, where a channel is a source-detector pair and
    wavelength is a separate axis. The short-channel count is omitted when the
    table carries no short-channel column, since the validator's check fires
    on the field's presence alone.

    Parameters
    ----------
    channel_table : pandas.DataFrame
        The channel table as written.

    Returns
    -------
    dict
        Channel counts for the sidecar.
    """
    counts = {
        'NIRSChannelCount': int(
            channel_table['type'].str.startswith('NIRS').sum()
        ),
    }

    if 'short_channel' in channel_table:
        counts['ShortChannelCount'] = int(
            (channel_table['short_channel'] == 'true').sum()
        )

    return counts


def _write_nirs_json(stream, prefix, task_name, channel_table,
                     scalp_labels=None, nirs_metadata=None):
    """
    Write the acquisition metadata sidecar.

    Fields come from three sources in increasing priority: those this writer
    computes, those the recording can answer for itself, and those the caller
    supplied.

    Parameters
    ----------
    stream : NirsStream
        Stream being described.
    prefix : str
        Filename stem to write beside.
    task_name : str
        Task label.
    channel_table : pandas.DataFrame
        The channel table, whose rows supply the channel counts.
    scalp_labels : pandas.DataFrame or None
        Nearest standard scalp positions, which fill the placement scheme.
        Written only when labelling was requested, since with a cap the cap
        fields describe the placement instead.
    nirs_metadata : dict, optional
        Caller-supplied fields, which override the rest.
    """
    sampling_rate = stream.data.attrs.get('sampling_rate')

    metadata = {
        "TaskName": task_name,
        # BIDS allows "n/a" here, but then channels.tsv MUST carry a
        # per-channel sampling_frequency column, so record what we have.
        "SamplingFrequency": (float(sampling_rate)
                              if sampling_rate is not None else "n/a"),
        "NIRSSourceOptodeCount": int(stream.probe.n_sources),
        "NIRSDetectorOptodeCount": int(stream.probe.n_detectors),
    }

    metadata.update(_channel_counts(channel_table))
    metadata.update(_derived_nirs_metadata(stream))

    if scalp_labels is not None:
        metadata["NIRSPlacementScheme"] = [
            str(position) for position in scalp_labels["position"]
        ]

    supplied = dict(nirs_metadata or {})
    unknown = set(supplied) - set(_NIRS_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown nirs.json field(s): {sorted(unknown)}. "
            f"Supported: {sorted(_NIRS_FIELDS)}. TaskName, SamplingFrequency "
            "and the three optode/channel counts are computed by the writer."
        )
    metadata.update(supplied)

    _write_json(metadata, f"{prefix}_nirs.json")


#: Column descriptions always written into the events sidecar. These are
#: properties of the BIDS columns themselves, not of any particular study, so
#: they need nothing from the caller.
_EVENT_COLUMNS = {
    "onset": {
        "LongName": "Event onset",
        "Description": "Onset of the event, measured from the start of the "
                       "acquisition of the first data point.",
        "Units": "s",
    },
    "duration": {
        "LongName": "Event duration",
        "Description": "Duration of the event, measured from onset.",
        "Units": "s",
    },
    "trial_type": {
        "LongName": "Event category",
        "Description": "Name of the condition this event belongs to.",
    },
}


def _write_events_tsv(stream, prefix, events_metadata=None):
    """
    Write the event table and its sidecar, if the stream carries events.

    Rows are sorted by onset at write time, since a file storing one group per
    condition yields events grouped rather than ordered.

    Parameters
    ----------
    stream : NirsStream
        Stream whose events are written.
    prefix : str
        Filename stem to write beside.
    events_metadata : dict, optional
        Extra sidecar fields.
    """
    if stream.events is None or stream.events.n_events == 0:
        return

    table = pd.DataFrame({
        "onset": stream.events.onsets,
        "duration": stream.events.durations,
        "trial_type": stream.events.labels,
    }).sort_values("onset", kind="stable").reset_index(drop=True)

    table.to_csv(f"{prefix}_events.tsv", sep="\t", index=False, na_rep="n/a")

    sidecar = dict(_EVENT_COLUMNS)
    if events_metadata:
        sidecar.update(events_metadata)

    _write_json(sidecar, f"{prefix}_events.json")


def _write_channels_tsv(stream, prefix, mode='strict'):
    """
    Write the channel table.

    One row per column of the stream's data time series, enumerated by the
    same call the SNIRF writer makes, so the rows and the data columns cannot
    fall out of step. The short-channel column is written only when the probe
    carries a threshold to classify by.

    Parameters
    ----------
    stream : NirsStream
        Stream being described.
    prefix : str
        Filename stem to write beside.
    mode : {'strict', 'compat', 'extended'}
        How to treat a modality BIDS has no channel type for.

    Returns
    -------
    pandas.DataFrame
        The table as written, whose rows supply the sidecar's channel counts.
    """
    probe = stream.probe

    short_mask = None
    if probe.sc_threshold is not None:
        try:
            short_mask = probe.get_short_channels()
        except ValueError:
            short_mask = None

    rows = []

    for column in describe_measurement_columns(stream):
        channel_type, units, note = _bids_channel_type(column, stream, mode)

        source = str(probe.source_labels[column.source_index - 1])
        detector = str(probe.detector_labels[column.detector_index - 1])

        # 'wavelength_nominal' is "n/a" for anything that is not raw NIRS;
        # a processed column carries a dataTypeLabel and no meaningful
        # wavelength index (see MeasurementColumn).
        if column.label is None:
            wavelength = str(int(probe.wavelengths[column.wavelength_index - 1]))
        else:
            wavelength = "n/a"

        row = {
            "name": f"{source}-{detector}",
            "type": channel_type,
            "source": source,
            "detector": detector,
            "wavelength_nominal": wavelength,
            "units": units,
        }

        if short_mask is not None:
            # BIDS spells booleans lowercase in TSVs, and its own check
            # compares against the string "true".
            row["short_channel"] = "true" if short_mask[column.channel_index] else "false"

        if note is not None:
            row["description"] = note

        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(f"{prefix}_channels.tsv", sep="\t", index=False, na_rep="n/a")
    return table


def _write_optodes_tsv(stream, prefix, geometry, scalp_labels=None):
    """
    Write the optode positions, in the resolved coordinate frame.

    Parameters
    ----------
    stream : NirsStream
        Stream being described.
    prefix : str
        Filename stem to write beside.
    geometry : dict
        Resolved positions and frame, from :func:`_resolve_geometry`.
    scalp_labels : pandas.DataFrame or None
        Nearest standard scalp positions, written to the optional description
        column. They go there rather than in the name column, which is the
        identity the channel table joins on and must stay unique.
    """
    probe = stream.probe

    source_positions = np.asarray(geometry['s_pos'])
    detector_positions = np.asarray(geometry['d_pos'])

    if source_positions.shape != (probe.n_sources, 3):
        raise ValueError("Source positions must have shape (n_sources, 3).")

    if detector_positions.shape != (probe.n_detectors, 3):
        raise ValueError("Detector positions must have shape (n_detectors, 3).")

    sources = pd.DataFrame({
        "name": probe.source_labels,
        "type": ["source"] * probe.n_sources,
        "x": source_positions[:, 0],
        "y": source_positions[:, 1],
        "z": source_positions[:, 2],
    })

    detectors = pd.DataFrame({
        "name": probe.detector_labels,
        "type": ["detector"] * probe.n_detectors,
        "x": detector_positions[:, 0],
        "y": detector_positions[:, 1],
        "z": detector_positions[:, 2],
    })

    table = pd.concat([sources, detectors], ignore_index=True)

    if scalp_labels is not None:
        described = {
            (row.name, row.type): f"{row.position} (nearest, {row.distance_mm:.0f} mm)"
            for row in scalp_labels.itertuples()
        }
        table["description"] = [
            described.get((name, kind), "n/a")
            for name, kind in zip(table["name"], table["type"])
        ]

    table.to_csv(f"{prefix}_optodes.tsv", sep="\t", index=False, na_rep="n/a")


#: BIDS accepts only these for *CoordinateUnits.
_COORDINATE_UNITS = {'m', 'mm', 'cm'}

COORDINATE_SYSTEMS = ('native', 'mni')


def _resolve_geometry(stream, coordinate_system):
    """
    Resolve the coordinate frame every spatial file for this stream is written
    in.

    Both the optode table and the coordinate-system sidecar are produced from
    the result, so the positions on disk and the frame declared beside them
    cannot disagree.

    Parameters
    ----------
    stream : NirsStream
        Stream being described.
    coordinate_system : {'native', 'mni'}
        'native' keeps the probe's own digitiser coordinates, declared as
        "Other" with a description, since the BIDS enumeration has no entry
        for an unknown frame. 'mni' transforms optodes and landmarks through
        the probe's landmark coregistration, recording the registration and
        its residuals.

    Returns
    -------
    dict
        Optode and landmark positions, the frame name, its units and a
        processing description.
    """
    probe = stream.probe

    if coordinate_system == 'native':
        units = probe.lengthUnit if probe.lengthUnit in _COORDINATE_UNITS else 'n/a'
        return {
            's_pos': np.asarray(probe.s_pos, dtype=float),
            'd_pos': np.asarray(probe.d_pos, dtype=float),
            'landmarks': probe.landmarks,
            'system': 'Other',
            'units': units,
            'description': (
                "Probe digitiser coordinates as recorded by the acquisition, "
                "not registered to any template. Axis orientation and origin "
                "are those of the digitiser used."
            ),
            'processing': 'none',
        }

    if probe.landmarks is None or probe.landmark_labels is None:
        raise ValueError(
            f"coordinate_system='mni' needs digitised anatomical landmarks to "
            f"register with, and stream {stream.name!r} has a probe with none. "
            "Use coordinate_system='native' to write the probe's own "
            "coordinates instead."
        )

    coreg = probe.coreg()
    residuals = coreg.residuals

    return {
        's_pos': coreg.mni_s_pos,
        'd_pos': coreg.mni_d_pos,
        'landmarks': coreg.mni_landmark_pos,
        'system': 'MNI152NLin2009aAsym',
        # coreg works in mm regardless of the probe's own unit.
        'units': 'mm',
        'description': None,
        'processing': (
            "Rigid-body (rotation and translation, no scaling) registration of "
            "the digitised anatomical landmarks onto the MNI152 reference "
            "fiducials. Per-landmark residuals (mm): "
            + ", ".join(f"{label} {error:.1f}" for label, error in residuals.items())
        ),
    }


def _write_coordsystem_json(prefix, geometry):
    """
    Write the coordinate-system sidecar.

    Parameters
    ----------
    prefix : str
        Filename stem to write beside.
    geometry : dict
        Resolved frame, from :func:`_resolve_geometry`.
    """
    selected_landmarks = {"NZ", "IZ", "RPA", "LPA", "CZ", "NAS", "NASION", "INION"}
    landmark_coordinates = {}

    labels = geometry.get('landmark_labels')
    positions = geometry.get('landmarks')

    if labels is not None and positions is not None:
        for label, position in zip(labels, positions):
            label = str(label).strip()
            if label.upper() in selected_landmarks:
                landmark_coordinates[label] = (
                    np.asarray(position).astype(float).tolist()
                )

    metadata = {
        "AnatomicalLandmarkCoordinateSystem": geometry['system'],
        "AnatomicalLandmarkCoordinateUnits": geometry['units'],
        "AnatomicalLandmarkCoordinates": landmark_coordinates,
        "NIRSCoordinateSystem": geometry['system'],
        "NIRSCoordinateUnits": geometry['units'],
        "NIRSCoordinateProcessingDescription": geometry['processing'],
    }

    # BIDS keeps two parallel groups: anatomical landmarks, and the fiducials
    # used to align the optodes. For fNIRS these are the same digitised points
    # -- Nz/LPA/RPA are both the anatomy and what the coregistration is fitted
    # to -- so both groups are written from them rather than leaving the
    # fiducial half empty.
    if landmark_coordinates:
        metadata.update({
            "FiducialsCoordinates": landmark_coordinates,
            "FiducialsCoordinateSystem": geometry['system'],
            "FiducialsCoordinateUnits": geometry['units'],
            "FiducialsDescription": (
                "Anatomical landmarks digitised with the optode positions, "
                "in the same coordinate system; these are the points the "
                "probe's coregistration is fitted to."
            ),
        })

    # REQUIRED whenever the system is "Other", and meaningless otherwise.
    if geometry['description'] is not None:
        metadata["AnatomicalLandmarkCoordinateSystemDescription"] = geometry['description']
        metadata["NIRSCoordinateSystemDescription"] = geometry['description']
        if landmark_coordinates:
            metadata["FiducialsCoordinateSystemDescription"] = geometry['description']

    _write_json(metadata, f"{prefix}_coordsystem.json")


# ---------------------------------------------------------------------------
# Session -> BIDS
# ---------------------------------------------------------------------------

def _iter_recordings(session):
    """
    Yield each stream to export, with its identifiers and entities.

    A session holding a participants mapping is flattened into those
    participants, each tagged with a recording entity.

    Parameters
    ----------
    session : Session
        Session to iterate.

    Yields
    ------
    tuple
        Subject, session identifier, the holder object, the stream name and
        its filename entities.
    """
    participants = getattr(session, 'participants', None)

    if participants:
        for role, child in participants.items():
            for stream_name in _stream_keys(child):
                entities = entities_from_stream_name(stream_name)
                entities.setdefault('recording', role)
                yield (getattr(session, 'dyad_id', child.subject_id),
                       session.session_id, child, stream_name, entities)
        return

    for stream_name in _stream_keys(session):
        yield (session.subject_id, session.session_id, session, stream_name,
               entities_from_stream_name(stream_name))


def _stream_keys(session):
    """Return every stream on a session, whether loaded or only registered."""
    keys = list(session.stream_paths.keys())
    keys += [name for name in session.streams if name not in session.stream_paths]
    return keys


def _resolve_stream(session, stream_name):
    """
    Return a stream and the file to copy for it.

    Parameters
    ----------
    session : Session
        Session holding the stream.
    stream_name : str
        Stream key.

    Returns
    -------
    tuple of (NirsStream, str or None)
        The stream, and the source path to copy, or None when it exists only
        in memory and must be written out.
    """
    entry = session.stream_paths.get(stream_name)
    source_path = entry['path'] if entry else None

    if source_path is not None and not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Stream {stream_name!r} of {session.subject_id} is registered at "
            f"{source_path}, which no longer exists."
        )

    # get_stream() honours the sc_threshold and events path recorded at
    # registration, so the exported sidecars describe the same probe the
    # rest of the analysis sees.
    stream = session.get_stream(stream_name)

    return stream, source_path


#: dataset_description.json fields a caller may supply. Name, BIDSVersion,
#: DatasetType and GeneratedBy are set by this writer and are not overridable.
_DATASET_FIELDS = (
    'License', 'Authors', 'Acknowledgements', 'HowToAcknowledge', 'Funding',
    'EthicsApprovals', 'ReferencesAndLinks', 'DatasetDOI', 'Keywords',
    'HEDVersion', 'SourceDatasets',
)


#: Keys BIDS allows in each SourceDatasets entry, all string-valued.
_SOURCE_DATASET_KEYS = ('URL', 'DOI', 'Version')


def _normalise_source_datasets(source_datasets):
    """
    Normalise the source-dataset provenance into the array BIDS expects.

    Accepts a URL or DOI string, a single mapping, or a list of either. Not
    derived from the exporting machine's own paths, which mean nothing to a
    reader.

    Parameters
    ----------
    source_datasets : str, dict, list or None
        Provenance as supplied.

    Returns
    -------
    list of dict
        One entry per source dataset.
    """
    if source_datasets is None:
        return None

    if isinstance(source_datasets, (str, dict)):
        entries = [source_datasets]
    else:
        entries = list(source_datasets)

    if not entries:
        raise ValueError(
            "source_datasets must name at least one source, or be omitted."
        )

    normalised = []
    for entry in entries:
        if isinstance(entry, str):
            # A bare string is a URL unless it looks like a bare DOI.
            key = 'DOI' if entry.lower().startswith('doi:') else 'URL'
            entry = {key: entry}

        unknown = set(entry) - set(_SOURCE_DATASET_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown SourceDatasets key(s): {sorted(unknown)}. "
                f"BIDS allows only {list(_SOURCE_DATASET_KEYS)}."
            )
        if not entry:
            raise ValueError(
                "Each source dataset needs at least one of "
                f"{list(_SOURCE_DATASET_KEYS)}."
            )

        normalised.append({key: str(value) for key, value in entry.items()})

    return normalised


def _validate_authors(authors):
    """
    Check that authors is a list of names rather than a single string.

    Parameters
    ----------
    authors : list of str
        Author names.

    Raises
    ------
    TypeError
        If a bare string is given, which would be read as one author per
        character.
    """
    if isinstance(authors, str):
        raise TypeError(
            f"authors must be a list of names, not a single string "
            f"({authors!r}). BIDS stores Authors as an array with one author "
            f'per entry -- pass ["{authors}"] for a single author.'
        )

    names = [str(author).strip() for author in authors]
    if not names or not all(names):
        raise ValueError(
            "authors must contain at least one non-empty name. BIDS's "
            "dataset_authors rule flags a dataset with no Authors, since it "
            "makes DOI registration from the dataset metadata impossible."
        )

    return names


def _build_dataset_description(dataset_name, bids_version, authors, license,
                               source_datasets, dataset_metadata):
    """
    Assemble the dataset description.

    A named argument overrides the same field supplied through
    ``dataset_metadata``. The licence is omitted rather than guessed when not
    given. The HED schema version is not written by default, since this writer
    produces no HED annotations.

    Parameters
    ----------
    dataset_name : str
        Dataset name.
    bids_version : str
        BIDS version to declare.
    authors : list of str
        Author names.
    license : str or None
        SPDX identifier.
    source_datasets : list of dict
        Normalised provenance.
    dataset_metadata : dict, optional
        Further fields.

    Returns
    -------
    dict
        The description.
    """
    from .. import __version__

    description = {
        "Name": dataset_name,
        "BIDSVersion": bids_version,
        "DatasetType": "raw",
        "GeneratedBy": [{
            "Name": "milob",
            "Version": __version__,
            "Description": "Organised into BIDS from SNIRF by milob's Study.to_bids().",
        }],
    }

    extra = dict(dataset_metadata or {})
    extra['Authors'] = _validate_authors(authors)

    if license:
        extra['License'] = str(license)

    sources = _normalise_source_datasets(source_datasets)
    if sources is not None:
        extra['SourceDatasets'] = sources

    unknown = set(extra) - set(_DATASET_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown dataset_description field(s): {sorted(unknown)}. "
            f"Supported: {sorted(_DATASET_FIELDS)}. Name, BIDSVersion, "
            "DatasetType and GeneratedBy are set by the writer."
        )

    for field in _DATASET_FIELDS:
        if field in extra:
            description[field] = extra[field]

    return description


#: Recorded in the README whenever a mode other than 'strict' was used, so
#: the dataset explains its own channels.tsv to whoever reads it next.
_MODE_NOTES = {
    'compat': (
        "## Channel types\n\n"
        "This dataset contains time-domain, frequency-domain or DCS fNIRS "
        "recordings. BIDS {version} defines six NIRS channel types and all "
        "six are continuous-wave, so these channels are typed `MISC` in "
        "`channels.tsv` with their real modality given in the `description` "
        "column. The `.snirf` files describe the data exactly and are the "
        "authoritative source; `NIRSChannelCount` is 0 for these runs "
        "because BIDS counts only rows carrying a NIRS type.\n"
    ),
    'extended': (
        "## Channel types\n\n"
        "This dataset contains time-domain, frequency-domain or DCS fNIRS "
        "recordings. BIDS {version} defines six NIRS channel types and all "
        "six are continuous-wave, so `channels.tsv` uses proposed type names "
        "(`NIRSTDGATEDAMPLITUDE`, `NIRSTDMOMENTAMPLITUDE`, "
        "`NIRSFDACAMPLITUDE`, `NIRSFDPHASE`, `NIRSDCSG2`, ...) that follow "
        "the specification's own naming but are **not part of it**. This "
        "dataset will therefore not pass the BIDS validator's channel-type "
        "check. The `.snirf` files are valid SNIRF and describe the data "
        "exactly.\n"
    ),
}


def _write_readme(output_path, dataset_name, readme, summary, mode='strict',
                  bids_version='1.10.0'):
    """
    Write the dataset README, which BIDS requires.

    A caller-supplied string is written verbatim; otherwise a stub is
    generated from what the export knows, with the parts only a person can
    supply marked as needing filling in.

    Parameters
    ----------
    output_path : str
        Dataset root.
    dataset_name : str
        Dataset name.
    readme : str or None
        Caller-supplied content.
    summary : dict
        Counts and task names gathered during the export.
    mode : str
        Channel-typing mode used.
    bids_version : str
        BIDS version declared.
    """
    note = ""
    if mode != 'strict' and summary.get('used_mode'):
        note = "\n" + _MODE_NOTES[mode].format(version=bids_version)

    if readme is not None:
        content = str(readme).rstrip() + "\n" + note
    else:
        from .. import __version__

        tasks = ", ".join(sorted(summary['tasks'])) or "not recorded"
        content = f"""# {dataset_name}

{summary['n_subjects']} participant(s), {summary['n_sessions']} session(s),
{summary['n_recordings']} fNIRS recording(s).

Task(s): {tasks}

## Description

TODO: describe the study -- what was measured, why, and under what protocol.

## Apparatus

TODO: describe the instrument, the probe layout, and the cap or montage used.

## Provenance

Organised into BIDS from SNIRF by milob {__version__}
(https://github.com/rcmesquita/milobnirs). The `.snirf` files are the raw
recordings; the accompanying `.tsv`/`.json` sidecars are derived from them.

This README was generated by that export and describes only what the files
themselves contain. Please replace the TODO sections above.
""" + note

    with open(os.path.join(output_path, "README"), "w", encoding="utf-8") as file:
        file.write(content)


def _anatomical_labels(stream, scalp_positions):
    """
    Compute the optional anatomical labels for one stream.

    Labelling needs digitised landmarks to place the probe in MNI space. A
    probe without them is not an error: every file is still written, without
    the extra description columns, and a warning names the stream affected.

    Parameters
    ----------
    stream : NirsStream
        Stream to label.
    scalp_positions : bool or {'10-20', '10-10'}
        Which standard position set to match against, or False to skip.

    Returns
    -------
    pandas.DataFrame or None
        Nearest scalp position per optode.
    """
    if not scalp_positions:
        return None

    probe = stream.probe
    if probe.landmarks is None or probe.landmark_labels is None:
        warnings.warn(
            f"Anatomical labelling was requested but stream {stream.name!r} has "
            "a probe with no digitised landmarks, so it cannot be placed in MNI "
            "space. Writing its optodes.tsv/channels.tsv without the "
            "description columns.",
            stacklevel=3,
        )
        return None

    scalp = None
    if scalp_positions:
        system = '10-10' if scalp_positions is True else scalp_positions
        scalp = probe.label_optodes_10_20(system=system)

    return scalp


def write_bids_dataset(sessions, output_path, *, dataset_name, authors,
                       overwrite=False, bids_version="1.10.0",
                       scalp_positions=False,
                       coordinate_system='native', license=None, readme=None,
                       source_datasets=None, dataset_metadata=None,
                       events_metadata=None, nirs_metadata=None,
                       mode='strict'):
    """
    Write sessions as a BIDS dataset.

    Writes the dataset description, README and participants table, and per
    recording the .snirf file, its metadata sidecar, and the events, channels,
    optodes and coordinate-system files. A stream registered as a path has its
    file copied; one held in memory is written out.

    Parameters
    ----------
    sessions : iterable of Session
        The recordings to export.
    output_path : str
        Destination directory, created if absent.
    dataset_name : str
        Dataset name.
    authors : list of str
        Author names, one per entry. Required by BIDS and not derivable from
        the data.
    overwrite : bool
        Replace a non-empty destination. Default False.
    bids_version : str
        BIDS version to declare.
    scalp_positions : bool or {'10-20', '10-10'}
        Label optodes with the nearest standard scalp position, written to the
        optode table's description column and the placement scheme. True means
        '10-10'. Requires digitised landmarks.
    coordinate_system : {'native', 'mni'}
        Frame the optode positions are written in. The frame is declared in
        the sidecar either way.
    license : str, optional
        SPDX identifier. Omitted when not given.
    readme : str, optional
        README content. A stub is generated when not given.
    source_datasets : str, dict or list, optional
        Provenance: a URL or DOI string, a mapping, or a list of either.
    dataset_metadata : dict, optional
        Further dataset-description fields.
    events_metadata : dict, optional
        Extra fields for each events sidecar.
    nirs_metadata : dict, optional
        Extra fields for each acquisition sidecar. Fields derivable from the
        recording are filled automatically and overridden by anything here.
    mode : {'strict', 'compat', 'extended'}
        How to treat a modality BIDS has no channel type for. 'strict'
        (default) refuses; 'compat' types it MISC with the real modality in
        the description; 'extended' writes proposed type names, which do not
        validate. The .snirf is written in full either way.

    Returns
    -------
    str
        The absolute output path.

    Raises
    ------
    TypeError
        If ``authors`` is not a list of names.
    ValueError
        If the destination is non-empty and ``overwrite`` is False, or if a
        stream cannot be represented under the chosen ``mode``.

    Notes
    -----
    Anatomical labels are descriptions rather than identities: an optode's
    name stays as the probe defines it, since that is what the channel table
    joins on. Labelling is off by default, being a derived quantity that
    depends on the coregistration.
    """
    sessions = list(sessions)
    if not sessions:
        raise ValueError(
            "No sessions to export. Populate the Study first (e.g. "
            "Study.load_bids_data(), Session.from_snirf(), or "
            "Study.add_session())."
        )

    output_path = os.path.abspath(output_path)

    if os.path.exists(output_path) and os.listdir(output_path) and not overwrite:
        raise FileExistsError(
            f"Destination is not empty: {output_path}. "
            "Use overwrite=True only if replacing its contents is intended."
        )

    if overwrite and os.path.exists(output_path):
        shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)

    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}.")

    if coordinate_system not in COORDINATE_SYSTEMS:
        raise ValueError(
            f"Unknown coordinate_system {coordinate_system!r}; "
            f"choose from {COORDINATE_SYSTEMS}."
        )

    _write_json(
        _build_dataset_description(dataset_name, bids_version, authors,
                                   license, source_datasets, dataset_metadata),
        os.path.join(output_path, "dataset_description.json"),
    )

    participant_rows = {}
    summary = {'tasks': set(), 'sessions': set(), 'n_recordings': 0,
               'used_mode': False}
    renames = {}
    written_prefixes = {}
    exported = 0

    for session in sessions:
        for subject, session_id, holder, stream_name, entities in _iter_recordings(session):
            stream, source_path = _resolve_stream(holder, stream_name)

            subject_label, session_label, entities = _sanitise_entities(
                subject, session_id, entities, renames
            )

            destination_dir = os.path.join(output_path, f"sub-{subject_label}")
            if session_label is not None:
                destination_dir = os.path.join(destination_dir, f"ses-{session_label}")
            destination_dir = os.path.join(destination_dir, "nirs")
            os.makedirs(destination_dir, exist_ok=True)

            prefix = os.path.join(
                destination_dir,
                _build_prefix(subject_label, session_label, entities),
            )
            # optodes.tsv / coordsystem.json describe the probe, which is
            # shared by every run in the session, so they carry no task
            # entity (BIDS inheritance resolves them for each run).
            spatial_prefix = os.path.join(
                destination_dir, _build_prefix(subject_label, session_label)
            )

            destination_snirf = f"{prefix}_nirs.snirf"

            # Two different stream names can sanitise to one label
            # ('n_back' and 'nBack' both give 'nBack'), and the second would
            # otherwise overwrite the first's files without a word.
            previous = written_prefixes.get(destination_snirf)
            if previous is not None:
                raise ValueError(
                    f"Streams {previous!r} and {stream_name!r} both map to "
                    f"{os.path.basename(destination_snirf)}. BIDS labels are "
                    "alphanumeric, so names differing only in punctuation or "
                    "capitalisation collide -- rename one of the streams."
                )
            written_prefixes[destination_snirf] = stream_name

            # channels.tsv annotates the columns of the SNIRF beside it, so
            # type it first: a stream whose modality BIDS cannot describe
            # then raises before any of its own files are written, rather
            # than leaving a .snirf with no valid table next to it.
            scalp = _anatomical_labels(stream, scalp_positions)

            channel_table = _write_channels_tsv(stream, prefix, mode=mode)

            # Did this stream actually need the non-strict handling? Only then
            # is the README note true of the dataset.
            if not channel_table['type'].str.startswith('NIRSCW').all():
                summary['used_mode'] = True

            if source_path is not None:
                shutil.copy2(source_path, destination_snirf)
            else:
                stream.to_snirf(destination_snirf)

            geometry = _resolve_geometry(stream, coordinate_system)
            geometry['landmark_labels'] = stream.probe.landmark_labels

            _write_nirs_json(stream, prefix, entities.get('task', stream_name),
                             channel_table, scalp_labels=scalp,
                             nirs_metadata=nirs_metadata)
            _write_events_tsv(stream, prefix, events_metadata=events_metadata)
            _write_optodes_tsv(stream, spatial_prefix, geometry,
                               scalp_labels=scalp)
            _write_coordsystem_json(spatial_prefix, geometry)

            stream.add_history('export_bids', {
                'path': os.path.relpath(destination_snirf, output_path),
                'entities': dict(entities),
                'scalp_positions': bool(scalp is not None),
            })

            summary['tasks'].add(entities.get('task', stream_name))
            summary['sessions'].add((subject_label, session_label))
            summary['n_recordings'] += 1

            metadata = dict(getattr(holder, 'metadata', None) or {})
            metadata.pop('participant_id', None)
            row = participant_rows.setdefault(subject_label, {})
            row.update(metadata)

            exported += 1

    if exported == 0:
        raise ValueError(
            "None of the given sessions hold any stream to export "
            "(no registered paths and no loaded streams)."
        )

    if renames:
        listed = ", ".join(f"{entity} {original!r} -> {label!r}"
                           for (entity, original), label in sorted(renames.items()))
        warnings.warn(
            "Some names were converted to BIDS labels, which are alphanumeric "
            f"('_' and '-' separate entities in a filename): {listed}. Pass "
            "names that are already alphanumeric to choose the labels "
            "yourself.",
            stacklevel=2,
        )

    _write_participants_tsv(participant_rows, output_path)

    summary['n_subjects'] = len(participant_rows)
    summary['n_sessions'] = len(summary['sessions'])
    if mode == 'extended' and summary['used_mode']:
        warnings.warn(
            "mode='extended' wrote channels.tsv with proposed NIRS type names "
            f"({', '.join(name for name, _ in _PROPOSED_CHANNEL_TYPES.values())}"
            ") that BIDS 1.10 does not define. This dataset will fail the "
            "validator's channel-type check and cannot be uploaded to "
            "OpenNeuro as it stands. Use mode='compat' for a dataset that "
            "validates. The rationale has been written into the README.",
            stacklevel=2,
        )

    _write_readme(output_path, dataset_name, readme, summary, mode=mode,
                  bids_version=bids_version)

    return output_path


def _write_participants_tsv(participant_rows, output_path):
    """
    Write the participants table, carrying whatever demographics the sessions
    hold.

    Session metadata is what a BIDS read parsed out of an incoming
    participants table, so preserving it closes the round trip. A session with
    no metadata contributes an identifier alone.

    Parameters
    ----------
    participant_rows : list of dict
        One row per participant.
    output_path : str
        Dataset root.
    """
    frame = pd.DataFrame([
        {"participant_id": f"sub-{subject}", **row}
        for subject, row in sorted(participant_rows.items())
    ])

    # BIDS spells a missing value "n/a" in a TSV, never an empty cell --
    # which is what a session lacking a column another session has would
    # otherwise leave behind.
    frame.to_csv(os.path.join(output_path, "participants.tsv"),
                 sep="\t", index=False, na_rep="n/a")
