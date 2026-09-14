import pandas as pd
import os
import re
import json
import warnings
import numpy as np
import xarray as xr
from scipy import stats
from glob import glob
from .. import __version__
from .session import Session
from .datastream import Datastream
from ..outputs.output_glm import GLMOutput

class Study:
    def __init__(self, name, data_path):
        self.name = name
        self.project_root = data_path
        self.derivatives_path = os.path.join('../results/milob_analysis')
        self.sessions = []


    def add_session(self, session):
        self.sessions.append(session)


    def to_bids(self, output_path, *, authors, overwrite=False,
                scalp_positions=False,
                coordinate_system='native', license=None, readme=None,
                source_datasets=None, dataset_metadata=None,
                events_metadata=None, nirs_metadata=None, mode='strict'):
        """
        Export this Study as a BIDS dataset.

        Writes dataset_description.json, README, participants.tsv, and per
        recording the .snirf, _nirs.json, _events.tsv, _events.json,
        _channels.tsv, _optodes.tsv and _coordsystem.json files. Driven by
        ``self.sessions``, falling back to discovering BIDS-named .snirf files in
        ``data_path`` when the Study holds none. Only continuous-wave recordings
        can be exported, since the BIDS channel-type vocabulary is entirely
        continuous-wave.

        Parameters
        ----------
        output_path : str
            Destination directory for the BIDS tree.
        authors : list of str
            Written to ``Authors`` in dataset_description.json, one name per
            entry. Required by BIDS and not derivable from the data.
        overwrite : bool
            Replace a non-empty destination. Default False.
        scalp_positions : bool or {'10-20', '10-10'}
            Label optodes with the nearest standard scalp position, written to
            the ``description`` column of optodes.tsv and to
            ``NIRSPlacementScheme``. True means '10-10'. Requires digitised
            landmarks.
        coordinate_system : {'native', 'mni'}
            Frame the optode positions are written in. 'native' keeps the
            digitiser coordinates; 'mni' registers them through each probe's
            landmarks. The frame is declared in coordsystem.json either way.
        license : str, optional
            SPDX identifier for dataset_description.json. Omitted when not given.
        readme : str, optional
            README content. A stub is generated when not given.
        source_datasets : str, dict, or list, optional
            Provenance written to ``SourceDatasets``: a URL or DOI string, a dict
            with any of URL, DOI and Version, or a list of either.
        dataset_metadata : dict, optional
            Further dataset_description.json fields, such as Acknowledgements,
            Funding or DatasetDOI.
        events_metadata : dict, optional
            Extra keys for each _events.json sidecar, notably
            ``StimulusPresentation``.
        nirs_metadata : dict, optional
            Extra keys for each _nirs.json sidecar, covering instrument, cap,
            institution and task protocol. Fields derivable from the recording
            are filled automatically and overridden by anything given here.
        mode : {'strict', 'compat', 'extended'}
            How to treat TD, FD and DCS streams. 'strict' (default) refuses;
            'compat' types them MISC with the real modality in ``description``;
            'extended' writes proposed type names, which do not validate. The
            .snirf is written in full either way.

        Returns
        -------
        str
            The output path.
        """
        from ..io.bids import write_bids_dataset

        sessions = self.sessions or self._sessions_from_flat_snirf()

        return write_bids_dataset(
            sessions,
            output_path,
            dataset_name=self.name,
            authors=authors,
            overwrite=overwrite,
            scalp_positions=scalp_positions,
            coordinate_system=coordinate_system,
            license=license,
            readme=readme,
            source_datasets=source_datasets,
            dataset_metadata=dataset_metadata,
            events_metadata=events_metadata,
            nirs_metadata=nirs_metadata,
            mode=mode,
        )


    def _sessions_from_flat_snirf(self):
        """Build Sessions from BIDS-named SNIRF files found directly in the data path."""
        sessions = {}

        for record in self._build_bids_records():
            key = (record["subject"], record["session"])

            if key not in sessions:
                sessions[key] = Session(subject_id=record["subject"],
                                        session_id=record["session"])

            sessions[key].add_stream_paths(
                record["stream_name"],
                record["source_path"],
                sc_threshold=None,
            )

        return [sessions[key] for key in sorted(sessions)]


    def _build_bids_records(self):
        """
        Parse flat SNIRF filenames into their BIDS destination metadata.

        Returns
        -------
        list of dict
            One record per file, carrying its path and BIDS entities.
        """
        snirf_files = sorted(glob(os.path.join(self.project_root, "*.snirf")))

        if not snirf_files:
            raise FileNotFoundError(
                f"No .snirf files found directly in: {self.project_root}"
            )

        pattern = re.compile(
            r"^sub-(?P<subject>[^_]+)"
            r"_ses-(?P<session>[^_]+)"
            r"_task-(?P<task>[^_]+)"
            r"(?P<optional_entities>(?:_(?:recording|run)-[^_]+)*)"
            r"_nirs\.snirf$"
        )

        records = []
        seen_destinations = set()

        for source_path in snirf_files:
            filename = os.path.basename(source_path)
            match = pattern.fullmatch(filename)

            if match is None:
                raise ValueError(
                    "Invalid filename for automatic BIDS organization: "
                    f"{filename}. Expected a name such as "
                    "'sub-01_ses-01_task-fingerTapping_nirs.snirf'."
                )

            destination_key = (
                match.group("subject"),
                match.group("session"),
                filename,
            )
            if destination_key in seen_destinations:
                raise ValueError(f"Duplicate BIDS destination detected: {filename}")

            seen_destinations.add(destination_key)
            records.append(
                {
                    "source_path": source_path,
                    "filename": filename,
                    "subject": match.group("subject"),
                    "session": match.group("session"),
                    "task": match.group("task"),
                    # Same key load_bids_data() would register this file
                    # under, so a flat folder and a loaded Study produce
                    # identical stream names -- and identical output names,
                    # since io.bids inverts this back into entities.
                    "stream_name": self._extract_bids_stream_name(filename),
                }
            )

        return records

    def load_bids_data(self, *, sc_threshold, filters=None,
                       events_suffix='_events.tsv'):
        """
        Crawl the data path for BIDS fNIRS data and populate ``sessions``.

        Parameters
        ----------
        sc_threshold : float or None
            Source-detector distance in mm below which a channel is short,
            recorded against every file as it is registered. Pass None if the
            probe design has no short-separation channels.
        filters : dict, optional
            Restrict what is loaded, e.g. ``{'sub': ['001', '002'], 'ses': ['01']}``,
            ``{'sex': 'f'}`` or ``{'age': ('between', (25, 40))}``.
        events_suffix : str
            Filename suffix of the events sidecar to register, replacing the
            ``_nirs.snirf`` ending. Default '_events.tsv'. A file with no matching
            sidecar falls back to the SNIRF's embedded stimulus data.
        """
        # Load participant metadata (participants.tsv)
        participants_file = os.path.join(self.project_root, 'participants.tsv')
        meta_df = None

        if os.path.exists(participants_file):
            try:
                meta_df = pd.read_csv(participants_file, sep='\t', encoding='utf-8')
            except:
                meta_df = pd.read_csv(participants_file, sep='\t', encoding='latin-1')
                
            # Ensure ID is is string and stripped of 'sub-'
            meta_df['participant_id'] = meta_df['participant_id'].str.replace('sub-','')


        # Find all subject directories (e.g., sub-001, sub-002, sub-01, ...)
        sub_dirs = sorted(glob(os.path.join(self.project_root, 'sub-*')))

        if not sub_dirs:
            print("No BIDS structure found.")

        for sub_path in sub_dirs:
            sub_id = os.path.basename(sub_path).replace('sub-', '')

            # Filter by subject ID or metadata (sex/age)
            if filters:
                # Check subject id filter
                if 'sub' in filters and sub_id not in filters['sub']:
                    continue
                # Check demographic filters (ages, sex, etc.)
                if meta_df is not None:
                    sub_meta = meta_df[meta_df['participant_id'] == sub_id]

                    skip_sub = False
                    for key, val in filters.items():
                        if key in sub_meta.columns:
                            meta_value = str(sub_meta[key].iloc[0])

                            # Numeric comparison operators (e.g. filter for people 40 and above (>=40))
                            if isinstance(val, tuple) and len(val) == 2:
                                op, threshold = val
                                
                                try:
                                    meta_value_num = float(meta_value)
                                except:
                                    continue

                                if op == '>=' and not (meta_value_num >= threshold):
                                    skip_sub = True
                                elif op == '<=' and not (meta_value_num <= threshold):
                                    skip_sub = True
                                elif op == '>' and not (meta_value_num > threshold):
                                    skip_sub = True
                                elif op == '<' and not (meta_value_num < threshold):
                                    skip_sub = True
                                elif op == 'between':
                                    low, high = threshold
                                    if not (low <= meta_value_num <= high):
                                        skip_sub = True
                                continue
                            # ---

                            if isinstance(val, (list, tuple, set)):
                                if meta_value not in [str(v) for v in val]:
                                    skip_sub = True
                            else:
                                if meta_value != str(val):
                                    skip_sub = True
                    if skip_sub:
                        continue
            
            # Extract subject metadata (always, regardless of filters)
            sub_meta_dict = {}
            if meta_df is not None:
                sub_meta_row = meta_df[meta_df['participant_id'] == sub_id]
                if not sub_meta_row.empty:
                    sub_meta_dict = sub_meta_row.iloc[0].to_dict()

            # Check for sessions (BIDS allows for optional sessions)
            ses_dirs = sorted(glob(os.path.join(sub_path, 'ses-*')))

            # If no ses- folders exist, treat the subject root as a single session
            if not ses_dirs:
                if filters and 'ses' in filters and "01" not in filters['ses']:
                    continue
                self._process_bids_nirs_folder(sub_path, sub_id, "01", filters=filters,
                                               subject_metadata=sub_meta_dict,
                                               sc_threshold=sc_threshold,
                                               events_suffix=events_suffix)
            else:
                for ses_path in ses_dirs:
                    ses_id = os.path.basename(ses_path).replace('ses-', '')

                    # Filter by session ID
                    if filters and 'ses' in filters and ses_id not in filters['ses']:
                        continue

                    self._process_bids_nirs_folder(ses_path, sub_id, ses_id, filters=filters,
                                                   subject_metadata=sub_meta_dict,
                                                   sc_threshold=sc_threshold,
                                                   events_suffix=events_suffix)

        import logging
        logging.getLogger('milob').info(f"Loaded {len(self.sessions)} sessions into Study: {self.name}")


    def _resolve_stream_names(self, fpath_basename_pairs):
        """
        Resolve stream keys for files registered together under one session.

        Base names shared by more than one file gain an ``_acq-<modality>``
        suffix, with the modality read from each file's SNIRF data-type codes, so
        two modalities filed under the same task do not collide.

        Parameters
        ----------
        fpath_basename_pairs : list of tuple of (str, str)
            Path and base stream name for each file.

        Returns
        -------
        dict of {str: str}
            Path to resolved stream name.
        """
        from collections import Counter
        from ..io.snirf import get_snirf_modality

        counts = Counter(name for _, name in fpath_basename_pairs)
        resolved = {}
        for fpath, base_name in fpath_basename_pairs:
            if counts[base_name] > 1:
                modality = get_snirf_modality(fpath).lower()
                resolved[fpath] = f"{base_name}_acq-{modality}"
            else:
                resolved[fpath] = base_name
        return resolved

    def _process_bids_nirs_folder(self, base_path, sub_id, ses_id, filters=None,
                                    subject_metadata=None,
                                    sc_threshold=None, events_suffix='_events.tsv'):
        """Register every SNIRF file in one BIDS subject or session folder."""
        nirs_path = os.path.join(base_path, 'nirs')
        if not os.path.exists(nirs_path):
            return

        # Find all .snirf files in this session
        snirf_files = glob(os.path.join(nirs_path, '*.snirf'))
        if not snirf_files:
            return

        # Apply filename filters up front so collision detection (below)
        # only looks at files that will actually be kept.
        kept = []
        for fpath in snirf_files:
            filename = os.path.basename(fpath)
            if self._file_passes_filters(filename, filters):
                kept.append((fpath, filename, self._extract_bids_stream_name(filename)))

        # ── standard mode: all streams on one Session ─────────────────── #
        session_obj = Session(subject_id=sub_id, session_id=ses_id)
        session_obj.metadata = subject_metadata or {}
        added_any_stream = False

        resolved_names = self._resolve_stream_names(
            [(fpath, stream_name) for fpath, _, stream_name in kept]
        )
        for fpath, filename, _ in kept:
            events_file = self._find_events_file(fpath, events_suffix)
            session_obj.add_stream_paths(resolved_names[fpath], fpath,
                                         sc_threshold=sc_threshold, events_path=events_file)
            added_any_stream = True

        if added_any_stream:
            self.sessions.append(session_obj)

    def _file_passes_filters(self, filename, filters):
        """Return True if a filename satisfies the given entity filters."""
        if not filters:
            return True

        if 'task' in filters:
            target_tasks = filters['task']
            if isinstance(target_tasks, str):
                target_tasks = [target_tasks]
            clean_targets = [t.replace('task-', '') for t in target_tasks]
            m = re.search(r'task-([^_]+)', filename)
            if (m.group(1) if m else None) not in clean_targets:
                return False

        if 'recording' in filters:
            target_recs = filters['recording']
            if isinstance(target_recs, str):
                target_recs = [target_recs]
            clean_recs = [r.replace('recording-', '') for r in target_recs]
            m = re.search(r'recording-([^_]+)', filename)
            if (m.group(1) if m else None) not in clean_recs:
                return False

        return True

    def _find_events_file(self, snirf_path, suffix='_events.tsv'):
        """
        Return the sidecar events path for a .snirf file, or None if absent.

        Parameters
        ----------
        snirf_path : str
            Path to the .snirf file.
        suffix : str
            Suffix replacing the file's ``_nirs.snirf`` ending.

        Returns
        -------
        str or None
        """
        prefix = snirf_path.replace('_nirs.snirf', '')
        candidate = prefix + suffix
        return candidate if os.path.exists(candidate) else None

    def _extract_bids_stream_name(self, filename):
        """
        Build a stream key from a BIDS filename.

        Combines the task, recording and run entities so that files sharing a task
        but differing by recording or run get distinct keys.

        Parameters
        ----------
        filename : str
            BIDS filename.

        Returns
        -------
        str
            Stream key, e.g. 'nback_run-1' or 'touch_recording-infant'.
        """
        task_match      = re.search(r'task-([^_]+)',      filename)
        recording_match = re.search(r'recording-([^_]+)', filename)
        run_match       = re.search(r'run-([0-9]+)',       filename)

        if not task_match:
            # No task entity in the filename. BIDS requires one for nirs
            # data, so the export has to write something; keep the fallback
            # alphanumeric so it is a usable label as it stands (see
            # io.bids.sanitise_label).
            return "nirs"

        name = task_match.group(1)

        if recording_match:
            name = f"{name}_recording-{recording_match.group(1)}"

        if run_match:
            name = f"{name}_run-{run_match.group(1)}"

        return name
    
    
    def create_group(self, group_name, filters):
        """
        Return a new Study holding only the sessions matching the filters.

        Filters against session metadata and identifiers already in memory; the
        filesystem is not rescanned.

        Parameters
        ----------
        group_name : str
            Label appended to the study name.
        filters : dict
            Same syntax as :meth:`load_bids_data`, e.g. ``{'group': 'MCI'}`` or
            ``{'age': ('>=', 40)}``.

        Returns
        -------
        Study
            Containing the matching sessions.
        """
        new_study = Study(name=f"{self.name}_{group_name}", data_path=self.project_root)
        new_study.sessions = [s for s in self.sessions if self._session_matches(s, filters)]
        return new_study


    def _session_matches(self, session, filters):
        """Return True if a session's metadata and identifiers satisfy the filters."""
        for key, val in filters.items():
            if key == 'sub':
                targets = [val] if isinstance(val, str) else list(val)
                if session.subject_id not in targets:
                    return False
                continue
            if key == 'ses':
                targets = [val] if isinstance(val, str) else list(val)
                if session.session_id not in targets:
                    return False
                continue

            if key not in session.metadata:
                return False

            meta_value = str(session.metadata[key])

            if isinstance(val, tuple) and len(val) == 2:
                op, threshold = val
                try:
                    meta_value_num = float(meta_value)
                except ValueError:
                    return False
                if op == '>=' and not (meta_value_num >= threshold):
                    return False
                elif op == '<=' and not (meta_value_num <= threshold):
                    return False
                elif op == '>' and not (meta_value_num > threshold):
                    return False
                elif op == '<' and not (meta_value_num < threshold):
                    return False
                elif op == 'between':
                    low, high = threshold
                    if not (low <= meta_value_num <= high):
                        return False
            elif isinstance(val, (list, set)):
                if meta_value not in [str(v) for v in val]:
                    return False
            else:
                if meta_value != str(val):
                    return False

        return True
    
    

    def load_all_streams(self):
        """Load every registered stream for every session."""
        for session in self.sessions:
            for name in session.stream_paths:
                session.get_stream(name)

    def check_probe_consistency(self, task_name, *, sc_threshold=None,
                                position_tol=1.0, reference='auto',
                                separation_range=None,
                                raise_on_mismatch=False, verbose=True):
        """
        Check whether one forward model or montage can serve every session.

        Reads geometry only, never a time series. Geometry is compared per
        channel by label, so a differing channel list is treated separately from
        optodes sitting in different places.

        Parameters
        ----------
        task_name : str
            Key in each session's ``stream_paths``.
        sc_threshold : float, optional
            Passed to ``Session.get_probe``. Defaults to each file's registered
            value.
        position_tol : float
            Largest optode displacement in mm still counted as the same montage.
        reference : str or 'auto'
            Subject whose montage to build on. 'auto' (default) picks the
            containing montage where one exists, otherwise the union. An explicit
            subject does not change the reported status.
        separation_range : tuple of (float, float), optional
            Source-detector separations in mm the analysis will use. When given,
            the report also says whether any differing channel falls inside that
            range.
        raise_on_mismatch : bool
            Raise ValueError rather than returning a report when the montages are
            incompatible. Default False.
        verbose : bool
            Print a summary of the status.

        Returns
        -------
        dict
            ``status`` is 'identical' (same channels and geometry), 'nested' (one
            session's channels contain every other's), 'overlapping' (geometry
            agrees but no session contains the rest, so the union is returned) or
            'incompatible' (geometry or wavelengths differ). Also carries
            ``consistent``, ``resolution``, ``reference``, ``reference_probe``,
            ``n_sessions``, ``n_channels``, ``common_channels``,
            ``union_channels``, ``differing_channels``,
            ``differing_separations_mm``, ``analysis_safe``, ``channels_at_risk``,
            ``geometry_ok``, ``mismatches`` and ``skipped``.

        Raises
        ------
        ValueError
            If the montages are incompatible and ``raise_on_mismatch`` is True.

        Examples
        --------
        >>> report = study.check_probe_consistency('fingertapping')
        >>> report['status']
        'nested'
        """
        from .probe import Probe

        if not self.sessions:
            raise ValueError("This study has no sessions to check.")

        probes, skipped = {}, {}
        for s in self.sessions:
            try:
                probes[s.subject_id] = s.get_probe(task_name,
                                                   sc_threshold=sc_threshold)
            except (KeyError, ValueError, OSError) as exc:
                skipped[s.subject_id] = str(exc)

        if not probes:
            raise ValueError(
                f"No session could supply a probe for task {task_name!r}. "
                f"Reasons: {skipped}"
            )

        labels = {sid: list(p.channel_labels) for sid, p in probes.items()}
        sets = {sid: set(v) for sid, v in labels.items()}
        union_set = set().union(*sets.values())
        common_set = set.intersection(*sets.values())

        # ── Geometry first: it decides whether anything else is worth
        #    asking. Compared against an arbitrary anchor, because
        #    agreement is transitive within a tolerance this coarse.
        anchor_id = next(iter(probes))
        geometry_ok, geometry_why = True, []
        for sid, probe in probes.items():
            if sid == anchor_id:
                continue
            rep = probes[anchor_id].compare(probe, position_tol=position_tol)
            if rep['max_optode_offset_mm'] is None or not rep['wavelengths_match'] \
                    or rep['max_optode_offset_mm'] > position_tol:
                geometry_ok = False
                geometry_why.append(f"sub-{sid}: {rep['summary']}")

        # ── Status, and the montage that resolves it ────────────────────
        containers = [sid for sid, v in sets.items() if v == union_set]
        if not geometry_ok:
            status = 'incompatible'
        elif len(common_set) == len(union_set):
            status = 'identical'
        elif containers:
            status = 'nested'
        else:
            status = 'overlapping'

        ref_probe, ref_id = None, None
        if reference != 'auto':
            if reference not in probes:
                raise KeyError(
                    f"reference={reference!r} is not among the sessions with a "
                    f"readable probe ({sorted(probes)})."
                )
            ref_id, ref_probe = reference, probes[reference]
        elif status == 'overlapping':
            # Session order, so the montage is stable across runs.
            ref_probe = Probe.union([probes[s.subject_id] for s in self.sessions
                                     if s.subject_id in probes],
                                    position_tol=position_tol)
        else:
            # Prefer a container that can actually be coregistered: a probe
            # without landmarks cannot seed `imaging.Coregistration`, and
            # picking one would move the failure somewhere less obvious.
            pool = containers or list(probes)
            ref_id = next((sid for sid in pool if probes[sid].landmarks is not None),
                          pool[0])
            ref_probe = probes[ref_id]

        differing = sorted(union_set - common_set)
        sep = dict(zip(ref_probe.channel_labels,
                       np.asarray(ref_probe.distances, dtype=float)))
        differing_sep = np.array([sep[c] for c in differing if c in sep])

        analysis_safe, at_risk = None, None
        if separation_range is not None:
            lo, hi = (float(x) for x in separation_range)
            short = ref_probe.sc_threshold
            at_risk = [c for c in differing
                       if c in sep and (lo <= sep[c] <= hi
                                        or (short is not None and sep[c] <= short))]
            analysis_safe = not at_risk

        mismatches = {}
        for sid, p in probes.items():
            if sid == ref_id:
                continue
            rep = ref_probe.compare(p, position_tol=position_tol)
            if not rep['match']:
                mismatches[sid] = rep

        resolution = {
            'identical': "one operator serves every session as built",
            'nested': (f"build the operator on sub-{ref_id}'s montage, then call "
                       "`operator.for_probe(stream, mask=...)` per session"),
            'overlapping': ("no session's montage contains the others; build on "
                            "report['reference_probe'] (a Probe.union), then call "
                            "`operator.for_probe(stream, mask=...)` per session"),
            'incompatible': ("build one operator per montage, or drop the "
                             "differing sessions"),
        }[status]

        if verbose:
            where = "union of all montages" if ref_id is None else f"sub-{ref_id}"
            print(f"probe consistency: {status} "
                  f"({len(probes)} session(s), reference {where}, "
                  f"{ref_probe.n_channels} channels)")
            if status != 'identical':
                print(f"  recorded by every session: {len(common_set):,} | "
                      f"by some only: {len(differing):,}", end='')
                if differing_sep.size:
                    print(f" ({differing_sep.min():.0f}-{differing_sep.max():.0f} mm)")
                else:
                    print()
            for line in geometry_why:
                print(f"  {line}")
            if analysis_safe is False:
                print(f"  WARNING: {len(at_risk)} differing channel(s) fall inside "
                      f"the analysis bands, e.g. {at_risk[:3]}")
            elif analysis_safe:
                print(f"  no differing channel is short or within "
                      f"{separation_range[0]}-{separation_range[1]} mm: "
                      "the analysis bands are identical across sessions")
            print(f"  -> {resolution}")
            for subject_id, why in skipped.items():
                print(f"  sub-{subject_id}: probe not read ({why})")

        if raise_on_mismatch and status == 'incompatible':
            raise ValueError(
                f"Sessions in this study do not share one montage. "
                + "; ".join(geometry_why) + f". {resolution}."
            )

        return {
            'status': status,
            'consistent': status != 'incompatible',
            'resolution': resolution,
            'reference': ref_id,
            'reference_probe': ref_probe,
            'n_sessions': len(probes),
            'n_channels': ref_probe.n_channels,
            'common_channels': sorted(common_set),
            'union_channels': sorted(union_set),
            'differing_channels': differing,
            'differing_separations_mm': differing_sep,
            'analysis_safe': analysis_safe,
            'channels_at_risk': at_risk,
            'geometry_ok': geometry_ok,
            'mismatches': mismatches,
            'skipped': skipped,
        }

    def get_group_stats(self, method='weighted', alpha=0.05):
        """
        Run a group-level analysis over the subject-level contrasts.

        Both methods are random-effects tests against the between-subject
        variance, with df = n_subjects - 1. Works on channel- and voxel-indexed
        outputs alike.

        Parameters
        ----------
        method : {'weighted', 'simple'}
            'simple' runs a one-sample t-test in which every subject counts
            equally. 'weighted' (default) runs a DerSimonian-Laird
            random-effects meta-analysis, weighting each subject by
            ``1 / (v_within + tau^2)``, and reduces to 'simple' when the
            subject-level variances are homogeneous.
        alpha : float
            Two-tailed significance threshold. Default 0.05.

        Returns
        -------
        GLMOutput
            Group-level estimates, standard errors, t-statistics and
            significance mask.
        """
        # Find all unique contrast names across all subjects
        all_contrast_names = set()
        for s in self.sessions:
            for res_obj in s.outputs.values():
                if 'contrast' in res_obj.output.dims:
                    all_contrast_names.update(res_obj.output.contrast.values)
        
        if not all_contrast_names:
            print("No contrast results found in any session.")
            return None
        
        # Run group analysis for each contrast
        group_results_list = []
        for con in sorted(list(all_contrast_names)):
            print(f"--- Running Group Analysis: {con} ---")
            
            all_betas = []
            all_vars = []
            subject_ids = []

            # Harvest data for specific contrast 'con'
            for s in self.sessions:
                for res_obj in s.outputs.values():
                    if 'contrast' in res_obj.output.dims and con in res_obj.output.contrast.values:
                        # Extract the slice
                        sub_con = res_obj.output.sel(contrast=con)
                        all_betas.append(sub_con.beta)
                        # `se` when compute_contrasts recorded it: p_var is
                        # (beta/t_stat)**2, which blows up to inf wherever a
                        # subject's t-stat came out ~0 -- harmless as a
                        # display quantity, poison as a weight.
                        if 'se' in sub_con.data_vars:
                            all_vars.append(sub_con.se ** 2)
                        else:
                            all_vars.append(res_obj.p_var.sel(contrast=con))
                        subject_ids.append(s.subject_id)
               
            if not all_betas:
                continue

            # Stack subjects into xarray: (subject, <channel|voxel>, <payload>)
            group_betas = xr.concat(all_betas, dim='subject').assign_coords(subject=subject_ids)
            group_vars = xr.concat(all_vars, dim='subject').assign_coords(subject=subject_ids)
            
            
            # Compute stats for contrast con
            if method == 'simple':
                res_ds = self._run_simple_group(group_betas, alpha)
            elif method == 'weighted':
                res_ds = self._run_weighted_group(group_betas, group_vars, alpha)
                
            # Add the contrast label back to the dataset:
            group_results_list.append(res_ds.expand_dims(contrast=[con]))
            
        # Merge all contrasts into one single Output object
        combined_group_ds = xr.concat(group_results_list, dim='contrast')
        
        # Find probe info -- Session has no single "session probe" of its
        # own (it can hold streams with genuinely different probes, e.g.
        # different modalities), so pull one from whichever loaded stream
        # has one, purely as a representative for group-level plotting.
        probe_info = None
        for s in self.sessions:
            for stream in getattr(s, 'streams', {}).values():
                if getattr(stream, 'probe', None) is not None:
                    probe_info = stream.probe
                    break
            if probe_info: break
            # Fallback: check if any output objects have it
            for res_obj in s.outputs.values():
                if res_obj.probe is not None:
                    probe_info = res_obj.probe
                    break
            if probe_info: break
        
        # Carry the voxel grid across from the subject outputs, so a group
        # map stays plottable on anatomy (GLMOutput.plot_voxel_3d).
        grid_info = None
        for s_ in self.sessions:
            for res_obj in s_.outputs.values():
                if getattr(res_obj, 'voxel_grid', None) is not None:
                    grid_info = res_obj.voxel_grid
                    break
            if grid_info is not None:
                break

        return GLMOutput(combined_group_ds, 
                      probe=probe_info, 
                      analysis_type=f"Group_{method.capitalize()}",
                      voxel_grid=grid_info)


    @staticmethod
    def _spatial_dim(Y_sub):
        """Spatial dimension of a stacked group array: 'channel' or 'voxel'."""
        for dim in ('channel', 'voxel'):
            if dim in Y_sub.dims:
                return dim
        raise ValueError(
            f"Group analysis needs 'channel' or 'voxel' as a spatial "
            f"dimension; got {Y_sub.dims}."
        )

    @staticmethod
    def _payload_dim(Y_sub):
        """Payload dimension of a stacked group array: 'chromophore', 'component' or 'wavelength'."""
        for dim in ('chromophore', 'component', 'wavelength'):
            if dim in Y_sub.dims:
                return dim
        raise ValueError(
            f"Group analysis needs one of 'chromophore'/'component'/"
            f"'wavelength' as a payload dimension; got {Y_sub.dims}."
        )

    def _run_weighted_group(self, Y_sub, Var_sub, alpha):
        """
        Run a DerSimonian-Laird random-effects group model.

        Each subject is weighted by ``1 / (v_within + tau^2)``, with tau^2 the
        between-subject variance. The group estimate and its standard error are
        the inverse-variance pair, giving ``t = beta / SE`` on
        ``df = n_subjects - 1``.

        Parameters
        ----------
        Y_sub : xr.DataArray
            Subject contrast estimates, with dims (subject, <spatial>, payload).
        Var_sub : xr.DataArray
            Within-subject variance of those estimates, same shape.
        alpha : float
            Two-tailed significance threshold.

        Returns
        -------
        dict
            Group estimate, standard error, t-statistic and significance mask.

        References
        ----------
        DerSimonian, R., & Laird, N. (1986). Controlled Clinical Trials, 7(3),
        177-188.
        """
        space = self._spatial_dim(Y_sub)
        y_raw = Y_sub.values
        v_raw = Var_sub.values

        # A subject contributes to a cell only with a finite estimate AND a
        # usable positive variance -- p_var can come back as inf/0 where a
        # subject-level t-stat was ~0, and either would poison the weights.
        valid = np.isfinite(y_raw) & np.isfinite(v_raw) & (v_raw > 0)
        n_sub = valid.sum(axis=0)

        y = np.where(valid, y_raw, 0.0)
        # Invalid cells get infinite variance, i.e. exactly zero weight, so
        # they drop out of every sum below without special-casing.
        v = np.where(valid, v_raw, np.inf)

        def _safe_div(num, den):
            return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float),
                             where=den > 0)

        with np.errstate(divide='ignore', invalid='ignore'):
            # ── DerSimonian-Laird tau^2, from the fixed-effect fit ──────
            w0 = np.where(valid, 1.0 / v, 0.0)
            sw0 = w0.sum(axis=0)
            beta_fe = _safe_div((w0 * y).sum(axis=0), sw0)
            Q = np.nansum(w0 * (y - beta_fe) ** 2, axis=0)
            # C = sum(w) - sum(w^2)/sum(w); zero when only one subject
            # contributes, which is why tau^2 falls back to 0 there.
            C = sw0 - _safe_div((w0 ** 2).sum(axis=0), sw0)
            tau_sq = np.where(C > 0, np.maximum(0.0, _safe_div(Q - (n_sub - 1), C)), 0.0)
            tau_sq = np.nan_to_num(tau_sq, nan=0.0)

            # ── random-effects weights and inference ───────────────────
            w = np.where(valid, 1.0 / (v + tau_sq[np.newaxis, ...]), 0.0)
            sw = w.sum(axis=0)
            beta = _safe_div((w * y).sum(axis=0), sw)
            se = np.sqrt(_safe_div(np.ones_like(sw), sw))
            t_stats = _safe_div(beta, se)

        df = np.maximum(n_sub - 1, 1)
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))

        enough = n_sub >= 2
        beta = np.where(n_sub > 0, beta, np.nan)
        for arr in (t_stats, p_vals, se):
            arr[~enough] = np.nan

        pdim = self._payload_dim(Y_sub)
        dims = [space, pdim]
        return xr.Dataset({
            'beta': (dims, beta),
            'se': (dims, se),
            't_stat': (dims, t_stats),
            'p_val': (dims, p_vals),
            'tau_sq': (dims, tau_sq),
            'n_subjects': (dims, n_sub),
            'is_significant': (dims, (p_vals < alpha) & enough),
        }, coords={space: Y_sub[space], pdim: Y_sub[pdim]})


    def _run_simple_group(self, Y_sub, alpha):
        """
        Run an unweighted one-sample t-test across subjects.

        Parameters
        ----------
        Y_sub : xr.DataArray
            Subject contrast estimates, with dims (subject, <spatial>, payload).
        alpha : float
            Two-tailed significance threshold.

        Returns
        -------
        dict
            Group estimate, standard error, t-statistic and significance mask.
        """
        import numpy as np
        from scipy import stats

        # Identify valid data points
        valid_mask = ~np.isnan(Y_sub.values)    # (subject, channel, chromophore)
        
        # Calculate N per channel/chromophore
        n_sub_per_cell = np.sum(valid_mask, axis=0)
        
        # Calculate Group Mean (Beta)
        group_beta = np.nanmean(Y_sub.values, axis=0)
        
        # Calculate Group Standard Deviation
        group_std = np.nanstd(Y_sub.values, axis=0, ddof=1)
        
        # Calculate T-Statistic: T = Mean / (Std / sqrt(N))
        standard_error = group_std / np.sqrt(n_sub_per_cell + 1e-10)    # Add epsilon to denominator to prevent division by zero
        t_stats = group_beta / (standard_error + 1e-10)
        
        # Degrees of Freedom and P-values
        df = n_sub_per_cell - 1
        df_clipped = np.maximum(df, 1)  # Set a floor of 1 for DF to avoid math errors (though T-test needs > 1)
        
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_clipped))
        
        # Mask out results where N < 2 (cannot do a T-test with 1 or 0 subjects)
        insufficient_data = n_sub_per_cell < 2
        t_stats[insufficient_data] = np.nan
        p_vals[insufficient_data] = np.nan

        # Package into Xarray Dataset
        space = self._spatial_dim(Y_sub)
        pdim = self._payload_dim(Y_sub)
        dims = [space, pdim]
        ds_group = xr.Dataset({
            'beta': (dims, group_beta),
            'se': (dims, standard_error),
            't_stat': (dims, t_stats),
            'p_val': (dims, p_vals),
            'n_subjects': (dims, n_sub_per_cell),
            'is_significant': (dims, p_vals < alpha)
        }, coords={space: Y_sub[space], pdim: Y_sub[pdim]})

        return ds_group
    

    def to_dataframe(self, output_key, level_names=('condition',)):
        """
        Collect every session's outputs into one long-form DataFrame.

        Each output type decides its own row structure. A session output that is
        a dict of outputs, as produced by segmented FC, is flattened with one
        added column per nesting level.

        Parameters
        ----------
        output_key : str
            Key in ``session.outputs`` to extract.
        level_names : tuple of str
            Column name per dict nesting level, outermost first. Default
            ``('condition',)``. Levels beyond the names given become 'level_2',
            'level_3' and so on.

        Returns
        -------
        pandas.DataFrame
            Long-form table carrying the output's own columns, one column per
            dict level, plus subject_id, session_id and each field of the
            session metadata.

        Examples
        --------
        >>> df = study.to_dataframe("glm_basic")
        >>> df.groupby(['condition', 'channel_i', 'channel_j'])['z'].mean()
        """
        import logging

        def _flatten(obj, depth=0):
            """Flatten a nested dict of outputs into (levels, output) pairs."""
            if hasattr(obj, 'to_dataframe'):
                return [obj.to_dataframe()]
            if isinstance(obj, dict):
                name = (level_names[depth] if depth < len(level_names)
                        else f'level_{depth}')
                out = []
                for key, value in obj.items():
                    for df in _flatten(value, depth + 1):
                        # Insert rather than assign, so the label lands at
                        # the front of the table where it reads as an index
                        # column rather than trailing the value columns.
                        df.insert(0, name, key)
                        out.append(df)
                return out
            if isinstance(obj, (list, tuple)):
                out = []
                for i, value in enumerate(obj):
                    for df in _flatten(value, depth + 1):
                        df.insert(0, (level_names[depth] if depth < len(level_names)
                                      else f'level_{depth}'), i)
                        out.append(df)
                return out
            raise TypeError(
                f"Study.to_dataframe: output '{output_key}' contains "
                f"{type(obj).__name__}, which has no to_dataframe() and is not "
                f"a dict/list of outputs."
            )

        rows = []
        missing = []
        for session in self.sessions:
            if output_key not in session.outputs:
                missing.append(session.subject_id)
                continue

            for df in _flatten(session.outputs[output_key]):
                df['subject_id'] = session.subject_id
                df['session_id'] = session.session_id
                for k, v in session.metadata.items():
                    df[k] = v
                rows.append(df)

        if missing:
            logging.getLogger('milob').warning(
                f"to_dataframe: output '{output_key}' missing for "
                f"{len(missing)} session(s): {missing}"
            )
        if not rows:
            raise ValueError(f"No sessions contain output '{output_key}'.")

        return pd.concat(rows, ignore_index=True)


    def get_group_connectivity(self, chromophore='HbT'):
        all_matrices = []
        for s in self.sessions:
            for res in s.outputs.values():
                if res.analysis_type == 'Connectivity':
                    # Extract matrix and apply Fisher Z: Z = arctanh(r)
                    r_matrix = res.output.sel(chromophore=chromophore).connectivity
                    z_matrix = np.arctanh(r_matrix.clip(-0.99, 0.99))
                    all_matrices.append(z_matrix)
        
        # Average in Z-space, then convert back to R-space
        avg_z = xr.concat(all_matrices, dim='subject').mean(dim='subject')
        avg_r = np.tanh(avg_z)
        
        return avg_r 



    def save(self):
        """Write the study configuration and trigger every session to save its results."""
        if not os.path.exists(self.derivatives_path):
            os.makedirs(self.derivatives_path)

        # 1. Ask every session to save its heavy .nc files
        for session in self.sessions:
            # Create sub-folder for the session results
            session_dir = os.path.join(self.derivatives_path, f"sub-{session.subject_id}", f"ses-{session.session_id}")
            session.save_results(session_dir)

        # 2. Save the Study JSON (The "Map")
        config_path = os.path.join(self.derivatives_path, 'study_config.json')

        session_entries = []
        for s in self.sessions:
            session_entries.append({
                "type": "session",
                "sub": s.subject_id,
                "ses": s.session_id,
                "results_files": list(s.outputs.keys()),
            })

        study_dict = {"study_name": self.name, "sessions": session_entries}
        with open(config_path, 'w') as f:
            json.dump(study_dict, f, indent=4)

    # ------------------------------------------------------------------ #
    # Multi-level FC / GLM                                                 #
    # ------------------------------------------------------------------ #

    def run_fc(self, preprocess_pipeline, method='pearson', average_runs=True,
               average_subjects=False, segment_events=False, event_labels=None,
               pad=(0.0, 0.0), min_duration=None,
               average_occurrences=True, reduce=None, roi_map=None,
               roi_kwargs=None, on_error='skip', **method_kwargs):
        """
        Compute functional connectivity across every session.

        Each session's own ``run_fc`` does the work; this adds the subject axis
        and per-session fault tolerance, logging and skipping a session that
        fails rather than aborting the batch.

        Parameters
        ----------
        preprocess_pipeline : list of tuple
            Preprocessing applied to every run of every session.
        method : str
            FC method. Default 'pearson'.
        average_runs : bool
            Average across runs within each session. Default True.
        average_subjects : bool
            Also average across sessions, returning a single result. Requires
            ``average_runs``. Default False.
        segment_events : bool
            Compute FC per event condition. Default False.
        event_labels : str or list of str, optional
            Restrict segmentation to these labels. A label absent from a session
            is omitted for that session.
        pad : tuple of (float, float)
            Seconds added before onset and after offset when segmenting.
        min_duration : float, optional
            Drop segments shorter than this many seconds.
        average_occurrences : bool
            Average repeated occurrences of a condition into one result.
            Default True.
        reduce : dict, optional
            Forwarded to ``FCOutput.reduce()`` per session. Strongly recommended
            for frequency-resolved methods across a large batch, since results
            otherwise accumulate at full resolution.
        roi_map : dict, Probe, or callable, optional
            Forwarded to ``FCOutput.roi_average()`` per session. A callable is
            invoked once per subject with that subject's own probe, so the
            assignment is recomputed rather than shared.
        roi_kwargs : dict, optional
            Options for the ROI averaging. Requires ``roi_map``.
        on_error : {'skip', 'raise'}
            'skip' (default) logs and skips a failing session; 'raise' re-raises
            the first failure.
        **method_kwargs
            Forwarded to ``FC.fit()``.

        Returns
        -------
        dict of {str: FCOutput} or FCOutput
            Keyed by subject, or a single result when averaging subjects.

        Raises
        ------
        ValueError
            If ``average_subjects`` is True without ``average_runs``.
        """
        from ..outputs.output_conn import FCOutput, check_fc_postprocess
        import logging

        log = logging.getLogger('milob')

        if average_subjects and not average_runs:
            raise ValueError("average_subjects=True requires average_runs=True.")

        # Validated here as well as inside each session's run_fc(): the
        # per-session try/except below turns any exception into a logged
        # "session skipped", so a typo'd roi_kwargs would otherwise surface
        # as every subject failing rather than as one clear error.
        check_fc_postprocess(roi_map, roi_kwargs, caller='Study.run_fc')

        subject_outputs = {}
        failed = {}
        for session in self.sessions:
            try:
                result = session.run_fc(
                    preprocess_pipeline=preprocess_pipeline, method=method,
                    average_runs=average_runs, segment_events=segment_events,
                    event_labels=event_labels, pad=pad, min_duration=min_duration,
                    average_occurrences=average_occurrences,
                    reduce=reduce, roi_map=roi_map, roi_kwargs=roi_kwargs,
                    **method_kwargs,
                )
            except Exception as e:
                if on_error == 'raise':
                    raise
                failed[session.subject_id] = f"{type(e).__name__}: {e}"
                log.warning(
                    f"Study.run_fc: session '{session.subject_id}' failed, skipping: "
                    f"{type(e).__name__}: {e}"
                )
                # Full traceback at DEBUG, so the default run stays readable
                # but the origin is recoverable without re-running: a message
                # alone can't tell a failure inside MILOB from one inside a
                # user-supplied callable (e.g. roi_map), which is exactly when
                # per-session skipping is most confusing.
                log.debug(f"Study.run_fc: traceback for '{session.subject_id}'", exc_info=True)
                continue

            session.outputs['fc'] = result
            subject_outputs[session.subject_id] = result

        if failed:
            log.warning(
                f"Study.run_fc: {len(failed)}/{len(self.sessions)} session(s) "
                f"failed and were skipped: {list(failed)}"
            )
        if not subject_outputs:
            raise ValueError("Study.run_fc: no sessions produced results.")

        if average_subjects:
            if segment_events:
                all_labels = {label for out in subject_outputs.values() for label in out}
                return {
                    label: FCOutput.average(
                        [out[label] for out in subject_outputs.values() if label in out]
                    )
                    for label in all_labels
                }
            return FCOutput.average(list(subject_outputs.values()))
        return subject_outputs

    def run_glm(self, preprocess_pipeline, glm_pipeline=None, average_runs=True,
                average_subjects=False, method='ar-irls'):
        """
        Run a GLM across every session.

        Parameters
        ----------
        preprocess_pipeline : list of tuple
            Preprocessing applied per run.
        glm_pipeline : list of tuple, optional
            GLM configuration steps.
        average_runs : bool
            Average across runs within each session. Default True.
        average_subjects : bool
            Average the per-session outputs into one group-level output, which
            requires contrasts to have been computed first. Default False.
        method : {'ols', 'robust', 'ar-irls'}
            Estimator for each session's GLM. Default 'ar-irls'.

        Returns
        -------
        dict of {str: GLMOutput} or GLMOutput
            Keyed by subject, or a single output when averaging subjects.
        """
        from ..outputs.output_glm import GLMOutput

        if average_subjects and not average_runs:
            raise ValueError("average_subjects=True requires average_runs=True.")

        subject_outputs = {}
        for session in self.sessions:
            result = session.run_glm(
                preprocess_pipeline=preprocess_pipeline,
                pipeline=glm_pipeline,
                average_runs=average_runs,
                method=method,
            )
            session.outputs['glm'] = result
            subject_outputs[session.subject_id] = result

        if average_subjects:
            return GLMOutput.average(list(subject_outputs.values()))
        return subject_outputs