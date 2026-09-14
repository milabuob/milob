

class Session:
    def __init__(self, subject_id: str = None, session_id: str = "01", name: str = None):
        self.subject_id = subject_id or "unknown"
        self.session_id = session_id
        self.name = name or f"Sub-{self.subject_id}_Ses-{session_id}"
        # {stream_name: {'path': ..., 'sc_threshold': ..., 'events_path': ...}}
        # -- registered but not necessarily loaded yet. sc_threshold/events_path
        # are recorded per-registration (set via add_stream_paths()) rather
        # than as a session-wide default, since no Probe exists yet at this
        # point for either to live on.
        self.stream_paths = {}
        self.streams = {}       # Dict to hold multiple datastreams (stays empty until requested)
        self.outputs = {}       # Dictionary to hold results for this session

        self.metadata = {}      # participant-level metadata from participants.tsv


    @classmethod
    def from_nirs(cls, filepath, *, sc_threshold, coord_file=None, length_unit="cm",
                  optical_name="nirs", aux_name="aux",
                  subject_id=None, session_id="01", name=None):
        """
        Load a .nirs file into a Session.

        The stimulus matrix is parsed into Events and attached to the returned
        stream.

        Parameters
        ----------
        filepath : str
            Path to the .nirs file.
        sc_threshold : float or None
            Source-detector distance in mm below which a channel is classified as
            short. Fixed on the Probe at construction. Pass None if the probe
            design has no short-separation channels.
        coord_file : str, optional
            AtlasViewer .txt file supplying 3-D probe geometry.
        length_unit : {'cm', 'mm'}
            Unit of the probe coordinates.
        optical_name : str
            Key for the optical stream. Default 'nirs'.
        aux_name : str
            Key for the auxiliary stream. Default 'aux'. Ignored when the file
            holds no aux data.
        subject_id : str, optional
            Subject identifier. Inferred from the filename stem if omitted.
        session_id : str
            Session identifier. Default '01'.
        name : str, optional
            Session name. Defaults to 'Sub-<subject_id>_Ses-<session_id>'.

        Returns
        -------
        Session
            Holding a CW_Stream and, where present, an AuxStream.
        """
        import os
        from ..io.nirs_mat import _parse_nirs_mat
        from .cw_nirs import CW_Stream
        from .auxiliary import AuxStream
        from .events import Events

        # Derive subject_id from filename if not given (e.g. "sub01.nirs" → "sub01")
        if subject_id is None:
            subject_id = os.path.splitext(os.path.basename(filepath))[0]

        parts = _parse_nirs_mat(filepath, coord_file=coord_file, length_unit=length_unit,
                                 sc_threshold=sc_threshold)

        session = cls(subject_id=subject_id, session_id=session_id, name=name)

        # Parse stimulus matrix into Events (if present)
        events = Events()
        if parts['stim_matrix'] is not None:
            events = Events.from_stim_matrix(parts['stim_matrix'], parts['times'])

        # Optical stream
        parts['cw_xr'].attrs['modality'] = 'cw_nirs'
        cw_stream = CW_Stream(parts['cw_xr'], parts['probe'], name=optical_name, events=events)
        session.add_stream(cw_stream)

        # Aux stream (only if the file contained aux data)
        if parts['aux_xr'] is not None:
            aux_stream = AuxStream(parts['aux_xr'], probe=None, name=aux_name)
            session.add_stream(aux_stream)

        return session


    @classmethod
    def from_nirx(cls, hdr_path, *, sc_threshold, probe_path=None, length_unit=None,
                  encoding=None, optical_name="nirs",
                  subject_id=None, session_id="01", name=None):
        """
        Load a NIRx dataset into a Session.

        NIRx stores intensity in per-wavelength text files alongside a header
        describing the acquisition. Probe geometry is not embedded and must be
        supplied separately.

        Parameters
        ----------
        hdr_path : str
            Path to the .hdr file. Sibling .wl1, .wl2 and further wavelength files
            must share its directory and filename stem.
        sc_threshold : float or None
            Source-detector distance in mm below which a channel is classified as
            short. Fixed on the Probe at construction. Pass None if the probe
            design has no short-separation channels.
        probe_path : str, optional
            Probe geometry file: a ``digpts.txt`` in AtlasViewer format, a
            ``.layout`` MATLAB file, or a Homer2 ``.SD`` file. Without it, optode
            positions are zero-filled and distances and topography are unavailable.
        length_unit : {'mm', 'cm'}, optional
            Unit of the probe coordinates. Inferred from the probe file extension
            when omitted.
        encoding : str, optional
            Text encoding of the header and digpts files. Read as UTF-8 and
            retried as latin-1 with a warning when omitted. Passing an explicit
            codec makes a mismatch raise instead.
        optical_name : str
            Key for the optical stream. Default 'nirs'.
        subject_id : str, optional
            Subject identifier. Inferred from the filename stem if omitted.
        session_id : str
            Session identifier. Default '01'.
        name : str, optional
            Session name. Defaults to 'Sub-<subject_id>_Ses-<session_id>'.

        Returns
        -------
        Session
            Holding a CW_Stream with events parsed from the stimulus matrix.
        """
        import os
        from ..io.nirx import _parse_nirx
        from .cw_nirs import CW_Stream
        from .events import Events

        if subject_id is None:
            subject_id = os.path.splitext(os.path.basename(hdr_path))[0]

        parts = _parse_nirx(hdr_path, probe_path=probe_path, length_unit=length_unit,
                            sc_threshold=sc_threshold, encoding=encoding)

        session = cls(subject_id=subject_id, session_id=session_id, name=name)

        events = Events()
        if parts['stim_matrix'] is not None:
            events = Events.from_stim_matrix(parts['stim_matrix'], parts['times'])

        parts['cw_xr'].attrs['modality'] = 'cw_nirs'
        cw_stream = CW_Stream(parts['cw_xr'], parts['probe'],
                              name=optical_name, events=events)
        session.add_stream(cw_stream)

        return session

    def to_snirf(self, path, stream_name='all', ml_format='indexed'):
        """
        Write this session's NIRS streams to SNIRF.

        Writing one stream writes it to ``path`` as a file. Writing several writes
        each to its own file inside ``path`` as a directory, named
        ``<session>_<stream>.snirf``, since a single SNIRF file cannot hold two
        modalities. Auxiliary streams are embedded in every file written.

        Parameters
        ----------
        path : str or Path
            Destination file when writing one stream, or directory when writing
            several. A directory is created if absent.
        stream_name : str, list of str, or 'all'
            Which streams to write. Default 'all', meaning every non-auxiliary
            NIRS stream.
        ml_format : {'indexed', 'array'}
            Encoding of the measurement list. Default 'indexed'.

        Returns
        -------
        str or dict of {str: str}
            The path written, or a mapping of stream name to path when writing
            more than one file.

        Raises
        ------
        ValueError
            If the session holds no NIRS streams, or ``stream_name`` does not name
            one.
        """
        import os
        from .auxiliary import AuxStream
        from ..io.snirf import write_snirf

        nirs_streams = {k: v for k, v in self.streams.items() if not isinstance(v, AuxStream)}
        aux_streams  = [v for v in self.streams.values() if isinstance(v, AuxStream)]

        if not nirs_streams:
            raise ValueError("Session has no NIRS streams to write.")

        if stream_name == 'all':
            names = list(nirs_streams)
        elif isinstance(stream_name, str):
            names = [stream_name]
        else:
            names = list(stream_name)

        missing = [n for n in names if n not in nirs_streams]
        if missing:
            raise ValueError(
                f"stream_name {missing} not found among session's NIRS streams: "
                f"{list(nirs_streams)}."
            )

        if len(names) == 1:
            write_snirf(nirs_streams[names[0]], path,
                        aux_streams=aux_streams or None, ml_format=ml_format)
            return path

        os.makedirs(path, exist_ok=True)
        written = {}
        for name in names:
            stream_path = os.path.join(path, f"{self.name}_{name}.snirf")
            write_snirf(nirs_streams[name], stream_path,
                        aux_streams=aux_streams or None, ml_format=ml_format)
            written[name] = stream_path
        return written

    @property
    def run_names(self):
        """Stream keys naming individual runs, which contain '_run-'."""
        return [k for k in self.stream_paths if '_run-' in k]

    @property
    def recording_names(self):
        """
        Stream keys naming individual recordings, which contain '_recording-'.

        In simultaneously-recorded datasets each session typically holds one
        stream per participant.
        """
        return [k for k in self.stream_paths if '_recording-' in k]

    @property
    def condition_names(self):
        """Stream keys produced by :meth:`segment_by_events`, which contain '_cond-'."""
        return [k for k in self.streams if '_cond-' in k]

    def add_stream_paths(self, stream_name, snirf_path, *, sc_threshold, events_path=None):
        """
        Register a SNIRF file for lazy loading by :meth:`get_stream`.

        Parameters
        ----------
        stream_name : str
            Key under which the stream will be loaded.
        snirf_path : str
            Path to the .snirf file. It is not read until the stream is requested.
        sc_threshold : float or None
            Source-detector distance in mm below which a channel is classified as
            short. Fixed on the Probe at construction. Pass None if the probe
            design has no short-separation channels.
        events_path : str, optional
            BIDS _events.tsv file to use instead of the embedded stimulus data.
        """
        self.stream_paths[stream_name] = {
            'path': snirf_path,
            'sc_threshold': sc_threshold,
            'events_path': events_path,
        }

    def add_stream(self, stream):
        """Add a stream to this session, keyed by its name."""
        # Using stream.name as the key allows for multiple tasks per session        
        self.streams[stream.name] = stream
        return self

    @classmethod
    def from_snirf(cls, filepath, *, sc_threshold, optical_name=None, aux_name="aux",
                   irf_name="irf",
                   subject_id=None, session_id="01", name=None,
                   external_events_path=None, **kwargs):
        """
        Load a SNIRF file into a Session.

        Parameters
        ----------
        filepath : str
            Path to the .snirf file.
        sc_threshold : float or None
            Source-detector distance in mm below which a channel is classified as
            short. Fixed on the Probe at construction. Pass None if the probe
            design has no short-separation channels.
        optical_name : str, optional
            Key for the primary NIRS stream. Defaults to the detected modality
            name.
        aux_name : str
            Key for the auxiliary stream. Default 'aux'. Ignored when the file
            holds no aux data.
        irf_name : str
            Key for a TD instrument-response stream. Default 'irf'. Loaded from a
            sibling ``*_{irf_name}.snirf`` file when exactly one exists, otherwise
            from an aux-label convention. Ignored when neither is present.
        subject_id : str, optional
            Subject identifier. Inferred from the filename stem if omitted.
        session_id : str
            Session identifier. Default '01'.
        name : str, optional
            Session name. Defaults to 'Sub-<subject_id>_Ses-<session_id>'.
        external_events_path : str, optional
            BIDS .tsv events file to use instead of the embedded stimulus data.

        Returns
        -------
        Session
            Holding the optical stream and, where present, auxiliary and IRF
            streams.
        """
        import os
        from glob import glob
        from .nirs import NirsStream
        from ..io.snirf import read_snirf_aux

        if subject_id is None:
            subject_id = os.path.splitext(os.path.basename(filepath))[0]

        session = cls(subject_id=subject_id, session_id=session_id, name=name)

        # Load NirsStream
        nirs_stream = NirsStream.from_snirf(
            filepath,
            name=optical_name,
            external_events_path=external_events_path,
            sc_threshold=sc_threshold,
            **kwargs
        )
        session.add_stream(nirs_stream)

        # IRF: the SNIRF spec has no dedicated IRF field -- an IRF is
        # physically nothing more than an ordinary TD measurement (dataType
        # 201 raw-gated / 301 moments, same timeDelays/timeDelayWidths/
        # momentOrders on /nirs/probe as any other TD stream), so prefer
        # loading it from its own sibling SNIRF file, the same per-stream
        # file layout Session.to_snirf() already uses for multi-stream
        # sessions (f"{session.name}_{stream_name}.snirf" next to the
        # primary file) -- this reuses the already-correct, already-spec-
        # compliant TD reader verbatim, no new spec bending. Checked before
        # (and independently of) the aux-based fallback below, since a
        # primary file with no aux data at all (the common case) would
        # otherwise never reach this check.
        #
        # Discovered by directory glob, not by reconstructing session.name --
        # Session.to_snirf() names sibling files f"{session.name}_{name}.snirf",
        # but `session.name` here may differ from whatever name the file was
        # originally written under (e.g. a caller loading without passing the
        # same `name=`/`subject_id=` back), so there is no reliable way to
        # rebuild that prefix from `filepath` alone. Globbing the primary
        # file's own directory for "*_{irf_name}.snirf" instead only assumes
        # one session's files share a directory (true for every file
        # `to_snirf()`'s multi-stream branch writes) -- ambiguous (>1 match)
        # or absent (0 match) results fall back to the aux convention below.
        irf_dir = os.path.dirname(filepath) or '.'
        irf_candidates = [
            p for p in glob(os.path.join(irf_dir, f"*_{irf_name}.snirf"))
            if os.path.abspath(p) != os.path.abspath(filepath)
        ]
        irf_loaded_from_sibling = len(irf_candidates) == 1
        if irf_loaded_from_sibling:
            irf_stream = NirsStream.from_snirf(
                irf_candidates[0], name=irf_name, sc_threshold=sc_threshold
            )
            session.add_stream(irf_stream)

        # Load Aux + (fallback) IRF Structures
        aux_dict = read_snirf_aux(filepath, data_xr=nirs_stream.data)

        if aux_dict is None:
            return session

        # Generic Aux -> AuxStream
        if aux_dict["generic"] is not None:
            from .auxiliary import AuxStream

            generic = aux_dict["generic"]

            if isinstance(generic, list):
                for i, da in enumerate(generic):
                    key = aux_name if i == 0 else f"{aux_name}_{i}"
                    session.add_stream(AuxStream(da, probe=None, name=key))
            else:
                session.add_stream(AuxStream(generic, probe=None, name=aux_name))

        # Non-spec Kernel DevKit aux-label IRF convention, only used if no
        # spec-compliant sibling file was found above.
        if not irf_loaded_from_sibling and aux_dict["irf"] is not None:
            from .td_nirs import TD_Stream

            irf_da = aux_dict["irf"]

            irf_stream = TD_Stream(
                irf_da,
                probe=nirs_stream.probe,
                name=irf_name,
                status="moment irf"
            )
            irf_stream.add_history('loaded_from_kernel_devkit_aux_convention', {
                'note': (
                    'Non-spec vendor extension (Kernel DevKit): IRF moments '
                    'embedded as /nirs/aux(j) groups labelled "irf-moments_*". '
                    'Prefer a sibling TD SNIRF file (see Session.to_snirf()) '
                    'for new data -- that path is spec-compliant and round-trips.'
                ),
            })

            session.add_stream(irf_stream)

        return session


    def get_probe(self, task_name, sc_threshold=None):
        """
        Read probe geometry from a registered SNIRF file without loading data.

        Parameters
        ----------
        task_name : str
            Key in ``stream_paths``, as used with :meth:`get_stream`.
        sc_threshold : float or None, optional
            Short-channel distance in mm for this call. Defaults to the value
            recorded when the file was registered.

        Returns
        -------
        Probe
            With channel labels, positions and, where the file carries them,
            landmarks.

        Examples
        --------
        >>> probe = session.get_probe('fingerTapping')
        >>> probe.add_roi('prefrontal', ['S1D1', 'S2D1'])
        """
        if task_name not in self.stream_paths:
            raise KeyError(
                f"Task '{task_name}' not found in this session. "
                f"Available: {list(self.stream_paths.keys())}"
            )
        from ..io.snirf import read_snirf_probe
        entry = self.stream_paths[task_name]
        effective_threshold = sc_threshold if sc_threshold is not None else entry['sc_threshold']
        return read_snirf_probe(entry['path'], sc_threshold=effective_threshold)

    def get_stream(self, name, sc_threshold=None):
        """
        Load a registered stream, reading its file on first access.

        Parameters
        ----------
        name : str
            Stream key, as registered by :meth:`add_stream_paths`.
        sc_threshold : float or None, optional
            Short-channel distance in mm for this call. Defaults to the value
            recorded when the file was registered.

        Returns
        -------
        Datastream
            The loaded stream, with any registered BIDS events attached.
        """
        if name not in self.streams:
            import logging
            from .nirs import NirsStream
            logging.getLogger('milob').info(f"Loading data for {self.subject_id} {self.session_id} {name}...")

            # Get registration info (path + defaults recorded at add_stream_paths() time)
            entry = self.stream_paths[name]
            snirf_path = entry['path']
            events_path = entry['events_path']

            effective_threshold = sc_threshold if sc_threshold is not None else entry['sc_threshold']

            # Build object
            self.streams[name] = NirsStream.from_snirf(
                snirf_path, name=name, external_events_path=events_path,
                sc_threshold=effective_threshold
            )

        return self.streams[name]

    def concatenate_streams(self, stream_names, output_name=None, preserve_gap=False):
        """
        Concatenate session streams and store the result.

        Streams must already be in optical-density state.

        Parameters
        ----------
        stream_names : list of str
            Ordered keys in ``streams`` to concatenate.
        output_name : str, optional
            Key under which the result is stored. Defaults to
            'concat_<name1>_<name2>_...'.
        preserve_gap : bool
            Keep any inter-run gap encoded in the time coordinates. Default False.

        Returns
        -------
        Datastream
            The concatenated stream, also stored in ``streams``.

        Examples
        --------
        >>> merged = session.concatenate_streams(["task1_od", "task2_od"])
        """
        from .utils import concatenate_streams as _concat

        streams = [self.get_stream(n) for n in stream_names]
        result = _concat(streams, name=output_name, preserve_gap=preserve_gap)
        self.add_stream(result)
        return result

    def segment_by_events(self, stream_input, labels=None, pad=(0.0, 0.0),
                           min_duration=None, register=True):
        """
        Split a session stream into per-condition segments.

        Preprocess the source stream fully before segmenting, since filtering a
        short slice behaves differently at its edges than filtering the whole
        recording and then slicing.

        Parameters
        ----------
        stream_input : str or Datastream
            Name of a loaded stream, or the stream itself.
        labels : str or list of str, optional
            Condition labels to extract. Defaults to every label present.
        pad : tuple of (float, float)
            Seconds added before onset and after offset.
        min_duration : float, optional
            Drop segments shorter than this many seconds.
        register : bool
            Add each segment to ``streams`` under a generated name. Default True.

        Returns
        -------
        dict of {str: list of Datastream}
            Segments per label, ordered by onset.

        Examples
        --------
        >>> od = session.preprocess('gonogo', pipeline=[('to_od', {}), ('tddr', {})])
        >>> segments = session.segment_by_events(od, labels=['rare', 'prevalent'])
        """
        stream = self.get_stream(stream_input) if isinstance(stream_input, str) else stream_input
        segments = stream.segment_by_events(labels=labels, pad=pad, min_duration=min_duration)

        if register:
            for seg_list in segments.values():
                for seg in seg_list:
                    self.add_stream(seg)

        return segments

    def unload_stream(self, name):
        """Drop a stream's data from memory, keeping its registered file path."""
        if name in self.streams:
            del self.streams[name]  
            

    def preprocess(self, stream_input, pipeline=None):
        """
        Run a sequence of preprocessing steps on a stream.

        Parameters
        ----------
        stream_input : str or Datastream
            Name of a registered stream, or the stream itself.
        pipeline : list of tuple
            Steps as (method_name, kwargs) pairs, e.g.
            ``[('to_od', {}), ('tddr', {'add_high_freq': True})]``.

        Returns
        -------
        Datastream
            The processed stream.
        """
                
        # Determine if we need to load or use the provided object
        if isinstance(stream_input, str):
            # User passed a name (e.g., "task-Hip")
            stream_name = stream_input
            stream = self.get_stream(stream_name)
        else:
            # User passed the actual stream object
            stream = stream_input
            stream_name = stream.name

        if pipeline is None:
            return stream

        # Execute the chain in pipeline
        import logging
        from .datastream import Datastream
        logging.getLogger('milob').info(f"--- Preprocessing {self.subject_id} : {stream_name} ---")
        for step_name, params in pipeline:
            # Get the method from the stream subclass (CW_Stream, TD_Stream, etc.)
            func = getattr(stream, step_name, None)

            if func is None:
                # Previously a printed warning that then carried on with the
                # step silently skipped -- i.e. data that is NOT preprocessed
                # the way the pipeline says it is, which every downstream
                # result then inherits. A typo'd step name is a bug in the
                # recipe, not something to continue past.
                import difflib
                available = [a for a in dir(stream)
                             if not a.startswith('_') and callable(getattr(stream, a, None))]
                suggestion = difflib.get_close_matches(step_name, available, n=3)
                raise AttributeError(
                    f"Pipeline step '{step_name}' is not a method of "
                    f"{type(stream).__name__}."
                    + (f" Did you mean: {suggestion}?" if suggestion else "")
                )

            # Execute the method (e.g., to_od, tddr)
            result = func(**params)

            # Every pipeline step must return a stream to chain from. Reporting
            # / inspection methods (roi_channel_summary -> DataFrame, plot_*
            # -> Figure) are valid methods, so getattr finds them happily, but
            # putting one in a pipeline replaces the stream with its return
            # value: the next step then fails somewhere unrelated, or -- if it
            # was the LAST step -- a DataFrame gets filed back into
            # session.streams as though it were preprocessed data.
            if not isinstance(result, Datastream):
                raise TypeError(
                    f"Pipeline step '{step_name}' returned "
                    f"{type(result).__name__}, not a Datastream. Only "
                    f"transforming steps belong in a pipeline; reporting and "
                    f"plotting methods (e.g. roi_channel_summary(), plot_*()) "
                    f"should be called on the preprocessed stream afterwards."
                )
            stream = result

        # Store the latest version back in the session dictionary
        self.streams[stream_name] = stream
        return stream
    

    def run_glm(self, stream_input=None, preprocess_pipeline=None, pipeline=None,
                average_runs=False, method='ar-irls', n_jobs=1):
        """
        Configure and fit a GLM.

        Pass ``stream_input`` for a single preprocessed stream, or
        ``preprocess_pipeline`` to preprocess and fit every run.

        Parameters
        ----------
        stream_input : str or Datastream, optional
            A single preprocessed stream. Mutually exclusive with
            ``preprocess_pipeline``.
        preprocess_pipeline : list of tuple, optional
            Preprocessing applied to every run.
        pipeline : list of tuple, optional
            GLM configuration steps, e.g.
            ``[('create_task_regressors', {'basis': 'canonical'})]``.
        average_runs : bool
            Average across runs into one output, which requires contrasts to have
            been computed on each. Default False.
        method : {'ols', 'robust', 'ar-irls'}
            Estimator for the GLM. Default 'ar-irls'.
        n_jobs : int
            Parallel jobs across channels within a run. Default 1.

        Returns
        -------
        GLMOutput or dict of {str: GLMOutput}
            A single output for one stream or when averaging runs, otherwise one
            per run.
        """
        import logging
        from ..analysis.glm import GLM
        from ..outputs.output_glm import GLMOutput

        def _fit_one(stream):
            model = GLM(stream)
            if pipeline:
                for step_name, step_params in pipeline:
                    func = getattr(model, step_name, None)
                    if func:
                        func(**step_params)
                    else:
                        print(f"Warning: GLM has no method '{step_name}'")
            logging.getLogger('milob').info(f"Fitting GLM for {self.subject_id}...")
            return model.fit(method=method, n_jobs=n_jobs)

        if preprocess_pipeline is None:
            # ── single-stream mode (backward compatible) ──────────────────────
            stream = self.get_stream(stream_input) if isinstance(stream_input, str) else stream_input
            return _fit_one(stream)

        # ── multi-run / multi-stream mode ────────────────────────────────────
        streams_to_process = self.run_names or list(self.stream_paths.keys())

        run_outputs = {}
        for stream_name in streams_to_process:
            processed = self.preprocess(stream_name, pipeline=preprocess_pipeline)
            run_outputs[stream_name] = _fit_one(processed)

        if not run_outputs:
            raise ValueError(f"No streams found for session {self.name}.")

        if average_runs:
            return GLMOutput.average(list(run_outputs.values()))
        return run_outputs


    def compute_fc(self, stream_input, method='pearson', **method_kwargs):
        """
        Compute functional connectivity for one preprocessed stream.

        Parameters
        ----------
        stream_input : str or Datastream
            Stream name in this session, or the stream itself.
        method : str
            FC method: 'pearson' (default), 'coherence', or 'wavelet_coherence'.
        **method_kwargs
            Forwarded to ``FC.fit()``, e.g. ``fs`` or ``wavelet``.

        Returns
        -------
        FCOutput
            The connectivity result.
        """
        from ..analysis.connectivity import FC

        stream = self.get_stream(stream_input) if isinstance(stream_input, str) else stream_input
        return FC(stream).fit(method=method, **method_kwargs)


    def run_fc(self, stream_input=None, preprocess_pipeline=None, method='pearson',
               average_runs=False, segment_events=False, event_labels=None,
               pad=(0.0, 0.0), min_duration=None, average_occurrences=True,
               reduce=None, roi_map=None, roi_kwargs=None, unload_after=True,
               **method_kwargs):
        """
        Compute functional connectivity for one stream or across all runs.

        Pass ``stream_input`` for a single stream, or ``preprocess_pipeline`` to
        preprocess and run every run. With ``segment_events``, FC is computed per
        event condition on the fully preprocessed stream.

        Parameters
        ----------
        stream_input : str or Datastream, optional
            A single stream. Mutually exclusive with ``preprocess_pipeline``.
        preprocess_pipeline : list of tuple, optional
            Preprocessing applied to every run.
        method : str
            FC method, forwarded to :meth:`compute_fc`. Default 'pearson'.
        average_runs : bool
            Average across runs into one result, pooling per label when
            segmenting. Default False.
        segment_events : bool
            Compute FC per event condition rather than over the whole run.
            Default False.
        event_labels : str or list of str, optional
            Restrict segmentation to these labels. A label absent from a run is
            omitted rather than raising. Used only with ``segment_events``.
        pad : tuple of (float, float)
            Seconds added before onset and after offset when segmenting.
        min_duration : float, optional
            Drop segments shorter than this many seconds.
        average_occurrences : bool
            Average repeated occurrences of a condition within a run into one
            result. Default True.
        reduce : dict, optional
            Forwarded to ``FCOutput.reduce()`` and applied to each result before
            averaging, e.g. ``{'freq': (0.012, 0.312)}``. Bounds memory for
            frequency-resolved methods and is a no-op for static ones.
        roi_map : dict, Probe, or callable, optional
            Forwarded to ``FCOutput.roi_average()`` and applied after ``reduce``.
            A callable is invoked once per run as ``roi_map(probe)`` with that
            run's preprocessed probe, and may return a dict, a Probe, or None if
            it defined the ROIs on the probe itself.
        roi_kwargs : dict, optional
            Options for the ROI averaging, such as 'agg', 'weights' and
            'weight_steepness'. Requires ``roi_map``.
        unload_after : bool
            Drop each run's preprocessed stream from memory once its FC is
            computed, bounding peak memory to one run. Default True. Ignored in
            single-stream mode.
        **method_kwargs
            Forwarded to ``FC.fit()``.

        Returns
        -------
        FCOutput or dict
            A single FCOutput for one stream or when averaging runs. Otherwise a
            dict keyed by run name, by condition label, or both when segmenting
            across multiple runs.

        Raises
        ------
        ValueError
            If ``roi_kwargs`` is given without ``roi_map``.

        Examples
        --------
        >>> fc = session.run_fc(preprocess_pipeline=pipeline, method='pearson')
        >>> per_condition = session.run_fc(preprocess_pipeline=pipeline,
        ...                                segment_events=True)
        """
        from ..outputs.output_conn import (
            FCOutput, apply_fc_postprocess, check_fc_postprocess, resolve_roi_map,
        )

        check_fc_postprocess(roi_map, roi_kwargs, caller='Session.run_fc')

        def _fit_stream(stream):
            # Resolved once per run, from THIS run's preprocessed probe --
            # not once for the whole call. A callable roi_map exists for
            # montages whose region assignment differs per subject (tile
            # placement varies with cap fitting), so it must see each
            # subject's own geometry; resolving here rather than inside
            # _post_process also means it runs once per run instead of once
            # per event occurrence.
            resolved_roi = resolve_roi_map(roi_map, stream, caller='Session.run_fc')

            def _post_process(result):
                return apply_fc_postprocess(result, reduce=reduce,
                                            roi_map=resolved_roi,
                                            roi_kwargs=roi_kwargs)

            if not segment_events:
                return _post_process(self.compute_fc(stream, method=method, **method_kwargs))

            segments = stream.segment_by_events(labels=event_labels, pad=pad, min_duration=min_duration)

            # Average within each label as soon as that label's occurrences
            # are computed, rather than building the full
            # {label: [occurrence, ...]} structure for EVERY label first
            # and only then averaging: a dict-comprehension-then-average
            # version
            # still computes and holds every occurrence of every condition
            # simultaneously before any of them are freed, so
            # average_occurrences=True would only shrink the FINAL result,
            # not the memory actually needed to get there. Each occurrence
            # is also reduced/roi-averaged (if requested) before joining
            # occ_results, for the same reason -- see reduce/roi_map
            # above.
            if average_occurrences:
                fc_by_label = {}
                for label, occs in segments.items():
                    occ_results = [
                        _post_process(self.compute_fc(seg, method=method, **method_kwargs))
                        for seg in occs
                    ]
                    fc_by_label[label] = FCOutput.average(occ_results)
                return fc_by_label

            return {
                label: [
                    _post_process(self.compute_fc(seg, method=method, **method_kwargs))
                    for seg in occs
                ]
                for label, occs in segments.items()
            }

        if preprocess_pipeline is None:
            # ── single-stream mode (backward compatible) ──────────────────────
            stream = self.get_stream(stream_input) if isinstance(stream_input, str) else stream_input
            return _fit_stream(stream)

        # ── multi-run / multi-stream mode ────────────────────────────────────
        # Prefer explicit run streams (_run-N); fall back to all stream paths
        # so that sessions with recording-based streams work naturally
        # without requiring the two-step preprocess→run_fc dance.
        streams_to_process = self.run_names or list(self.stream_paths.keys())

        run_outputs = {}
        for stream_name in streams_to_process:
            processed = self.preprocess(stream_name, pipeline=preprocess_pipeline)
            try:
                run_outputs[stream_name] = _fit_stream(processed)
            finally:
                if unload_after:
                    self.unload_stream(stream_name)

        if not run_outputs:
            raise ValueError(f"No streams found for session {self.name}.")

        if average_runs:
            if segment_events:
                # Pool every run's occurrences per label (not average-of-averages),
                # so average_occurrences=False upstream still averages correctly here.
                all_labels = {label for per_run in run_outputs.values() for label in per_run}
                pooled = {}
                for label in all_labels:
                    items = []
                    for per_run in run_outputs.values():
                        val = per_run.get(label)
                        if val is None:
                            continue
                        items.extend(val if isinstance(val, list) else [val])
                    pooled[label] = FCOutput.average(items)
                return pooled
            return FCOutput.average(list(run_outputs.values()))
        return run_outputs
                
        
        
    def save_results(self, folder_path=None):
        """
        Write this session's outputs and streams to NetCDF.

        Parameters
        ----------
        folder_path : str
            Destination directory.
        """
        import os
        
        if folder_path is None:
            folder_path = "./results"
            
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save GLM Results (The 'Output' objects)
        for name, result_obj in self.outputs.items():
            # Filename: sub-001_ses-01_glm_task-nback.nc
            fname = f"{self.subject_id}_{self.session_id}_{name}.nc"
            filepath = os.path.join(folder_path, fname)
            
            # Call the save method defined in Output class
            result_obj.save(filepath)
            
            

    def __repr__(self):
        return f"<Session | {self.name} | Streams: {list(self.streams.keys())}>"
         

    def fit_joint(self, fd_stream, dcs_stream, *, fd_sigma, dcs_sigma,
                  wavelengths=None, param_config=None, fixed_params=None,
                  n=1.33, freq=None, output_name=None, time_index=0,
                  n_starts=None, n_jobs=1, store=True):
        """
        Jointly fit FD-DOS and CW-DCS streams against one shared composition.

        Both modalities constrain the same HbO, HbR and scattering parameters, and
        absorption and scattering follow from them at each block's wavelength. The
        DCS laser line therefore need not be one of the FD wavelengths, and
        composition uncertainty propagates into blood flow.

        Parameters
        ----------
        fd_stream, dcs_stream : str or Datastream
            Stream names in this session, or the streams themselves.
        fd_sigma, dcs_sigma : float
            Measurement noise for each block. Required, since least squares
            otherwise weights blocks by point count and units, which misstates the
            reported uncertainty. ``fd_sigma`` applies to the transformed FD
            residual and ``dcs_sigma`` to g2(tau).
        wavelengths : float or sequence of float, optional
            Where to evaluate the returned optical properties. Defaults to the DCS
            stream's wavelengths.
        param_config : dict, optional
            Parameters to fit and their bounds. Defaults to
            :data:`~milob.forward.spectral.TISSUE_DCS_PARAM_CONFIG`.
        fixed_params : dict, optional
            Parameters held at a known value rather than fitted.
        n : float
            Refractive index.
        freq : float, optional
            FD modulation frequency. Resolved from the stream when omitted.
        output_name : str, optional
            Base name for the two result streams, stored as ``<name>_comp`` and
            ``<name>_op``. Defaults to '<fd_name>_<dcs_name>'.
        time_index : int
            Which time point to fit. One fit per call.
        n_starts : int, optional
            Number of multi-start attempts.
        n_jobs : int
            Parallel jobs for the fit.
        store : bool
            Add both result streams to this session. Default True; they are
            returned either way.

        Returns
        -------
        processing.fitting.JointFitResult
            A (tissue, optical) pair. Both carry the same covariance, whose param
            axis spans every fitted parameter across both spaces.

        Examples
        --------
        >>> tissue, optical = session.fit_joint("fd", "dcs",
        ...                                     fd_sigma=0.01, dcs_sigma=0.005)
        >>> optical.bfi
        """
        import numpy as np

        from ..forward.dcs import si_dcs_g1
        from ..forward.dos import si_fd_fluence
        from ..forward.spectral import (TISSUE_DCS_PARAM_CONFIG,
                                        assemble_spectral_dcs,
                                        assemble_spectral_fd, extinction_at)
        from ..processing.fitting import dcs_block, fd_block, fit_joint

        fd = self.get_stream(fd_stream) if isinstance(fd_stream, str) else fd_stream
        dcs = self.get_stream(dcs_stream) if isinstance(dcs_stream, str) else dcs_stream

        if param_config is None:
            param_config = TISSUE_DCS_PARAM_CONFIG

        freq = fd._resolve_freq(freq)
        fd_distances = fd.data.coords['distance'].values
        if fd.data.attrs.get('lengthUnit') == 'mm':
            fd_distances = fd_distances / 10.0
        fd_distances = np.asarray(fd_distances, dtype=float)

        fd_wavelengths = fd.data.wavelength.values
        if len(fd_wavelengths) < 2:
            raise ValueError(
                f"A joint spectral fit needs at least 2 FD wavelengths to "
                f"identify composition and scattering shape -- got "
                f"{len(fd_wavelengths)}. Use fit_to_op() + fit_to_bfi() "
                f"(the two-step workflow) for single-wavelength FD data."
            )

        blocks = []
        complex_arr = fd.data.sel(freq=freq).values      # (time, channel, wl)
        for wi, wl in enumerate(fd_wavelengths):
            eps_hbo, eps_hbr = extinction_at(float(wl))
            blocks.append(fd_block(
                fd_distances,
                # Same conjugation as FD_Stream.fit_to_op -- see its comment
                # on the phase convention.
                np.conj(complex_arr[time_index, :, wi]),
                si_fd_fluence,
                {"n": n, "wavelength": float(wl), "freq": freq,
                 "eps_hbo": eps_hbo, "eps_hbr": eps_hbr},
                assemble=assemble_spectral_fd, sigma=fd_sigma,
                name=f"fd_{int(wl)}"))

        dcs_distances = dcs.data.coords['distance'].values
        if dcs.data.attrs.get('lengthUnit') == 'mm':
            dcs_distances = dcs_distances / 10.0
        dcs_wavelengths = dcs.data.wavelength.values
        dcs_arr = dcs.data.values                        # (time, ch, wl, tau)

        for ci in range(len(dcs.data.channel)):
            for wi, wl in enumerate(dcs_wavelengths):
                eps_hbo, eps_hbr = extinction_at(float(wl))
                blocks.append(dcs_block(
                    dcs.taus, dcs_arr[time_index, ci, wi, :], si_dcs_g1,
                    {"rho": float(dcs_distances[ci]), "n": n,
                     "wavelength": float(wl),
                     "eps_hbo": eps_hbo, "eps_hbr": eps_hbr},
                    assemble=assemble_spectral_dcs, sigma=dcs_sigma,
                    observation="siegert" if dcs.observation == "g2" else "identity",
                    name=f"dcs_{ci}_{int(wl)}"))

        if wavelengths is None:
            wavelengths = dcs_wavelengths

        result = fit_joint(
            blocks, param_config, wavelengths=wavelengths,
            fixed_params=fixed_params, events=fd.events,
            history=fd.history, n_starts=n_starts, n_jobs=n_jobs)

        base = output_name or f"{fd.name}_{dcs.name}"
        if result.tissue is not None:
            result.tissue.name = f"{base}_comp"
            result.tissue.add_history('session_fit_joint', {
                'fd_stream': fd.name, 'dcs_stream': dcs.name,
                'fd_sigma': fd_sigma, 'dcs_sigma': dcs_sigma,
                'n_blocks': len(blocks), 'time_index': time_index})
        if result.optical is not None:
            result.optical.name = f"{base}_op"
            result.optical.add_history('session_fit_joint', {
                'fd_stream': fd.name, 'dcs_stream': dcs.name,
                'fd_sigma': fd_sigma, 'dcs_sigma': dcs_sigma,
                'n_blocks': len(blocks), 'time_index': time_index})

        if store:
            # Same convention as concatenate_streams()/preprocess(): file the
            # derived streams back into this session AND return them. The
            # '_comp'/'_op' suffixes keep them from colliding with each other
            # (add_stream keys on stream.name) or with their sources.
            for stream in (result.tissue, result.optical):
                if stream is not None:
                    self.add_stream(stream)

        return result

    def fit_bfi_from_dos(
        self, dos_stream, dcs_stream, *,
        dos_window='auto', gap_factor=5.0, mismatch_factor=5.0,
        dos_method='slope', dos_kwargs=None,
        wavelength='interp', time='interp',
        n=1.33,
        param_config=None, fixed_params=None, n_starts=None, n_jobs=1,
        dcs_kwargs=None,
        output_name=None, store=True,
    ):
        """
        Fit blood flow from DCS using optical properties measured by FD-DOS.

        Windows and averages the DOS stream, fits it for optical properties, maps
        those onto the DCS wavelengths and timestamps, and fits the DCS stream with
        them held fixed. Optical properties are treated as exact, so their
        uncertainty does not reach the flow error bar; :meth:`fit_joint` propagates
        it. DOS and DCS are assumed co-located.

        Parameters
        ----------
        dos_stream, dcs_stream : str or Datastream
            Stream names in this session, or the streams themselves. The first
            must be an FD_Stream and the second a DCS_Stream.
        dos_window : {'auto', None, 'events'}, float, or array-like
            How to window the DOS stream before fitting. 'auto' (default) splits
            wherever a frame gap exceeds ``gap_factor`` times the median sampling
            interval and coherently averages each block. None never averages, a
            float gives a fixed boxcar width in seconds, an array gives explicit
            bin edges, and 'events' splits at event onsets.
        gap_factor : float
            Multiple of the median sampling interval above which 'auto' splits.
            Default 5.
        mismatch_factor : float
            When 'auto' finds no gap, warn if the DOS and DCS frame counts differ
            by more than this ratio. Default 5.
        dos_method : {'slope', 'fit'}
            Which optical-property fit to run on the DOS stream. Default 'slope'.
        dos_kwargs : dict, optional
            Extra arguments forwarded to the DOS fit.
        wavelength : {'interp', 'nearest'}
            How DOS optical properties are mapped onto the DCS wavelengths.
            Default 'interp', falling back to 'nearest' with a warning below two
            DOS wavelengths.
        time : {'interp', 'nearest'}
            How DOS optical properties are aligned to the DCS timestamps. Default
            'interp'.
        n : float
            Refractive index passed to the DCS forward model. Default 1.33.
        param_config : dict, optional
            Parameters to fit and their bounds, forwarded to the DCS fit.
        fixed_params : dict, optional
            Parameters held at a known value rather than fitted.
        n_starts : int, optional
            Number of multi-start attempts.
        n_jobs : int
            Parallel jobs for the DCS fit.
        dcs_kwargs : dict, optional
            Extra arguments forwarded to the DCS fit. A ``forward_parameters``
            entry is merged on top of the mapped optical properties.
        output_name : str, optional
            Name for the returned stream. Defaults to '<dos>_<dcs>_bfi'.
        store : bool
            Add the returned stream, and the intermediate DOS result, to this
            session. Default True.

        Returns
        -------
        OptPropStream
            On the DCS stream's time axis. The op axis carries ``bfi`` alongside
            the optical properties used for each time point, ``uncertainty``
            carries the one-sigma from each fit, and ``covariance`` spans only the
            DCS-fitted parameters.

        Examples
        --------
        >>> bfi = session.fit_bfi_from_dos("dos", "dcs")
        >>> bfi.select("bfi")
        """
        import numpy as np
        import xarray as xr

        from .fd_nirs import FD_Stream
        from .dcs_stream import DCS_Stream
        from .opt_prop_stream import OptPropStream, _base_label

        dos = self.get_stream(dos_stream) if isinstance(dos_stream, str) else dos_stream
        dcs = self.get_stream(dcs_stream) if isinstance(dcs_stream, str) else dcs_stream

        if not isinstance(dos, FD_Stream):
            raise TypeError(f"dos_stream must be an FD_Stream, got {type(dos).__name__}.")
        if not isinstance(dcs, DCS_Stream):
            raise TypeError(f"dcs_stream must be a DCS_Stream, got {type(dcs).__name__}.")

        dos_kwargs = dict(dos_kwargs or {})
        dcs_kwargs = dict(dcs_kwargs or {})

        dcs_times = np.asarray(dcs.data.time.values, dtype=float)
        dcs_wls = np.asarray(dcs.data.wavelength.values, dtype=float)
        n_dcs_ch = dcs.data.sizes['channel']

        # --- stage 1: window / block-average the raw DOS -----------------
        dos_windowed, window_info = _fd_block_average(
            dos, dos_window, gap_factor=gap_factor,
            mismatch_factor=mismatch_factor, n_dcs=len(dcs_times))

        # --- stage 2: DOS optical-property fit --------------------------
        if dos_method == 'slope':
            op_dos = dos_windowed.slopefit_to_op(n=n, **dos_kwargs)
        elif dos_method == 'fit':
            op_dos = dos_windowed.fit_to_op(n=n, **dos_kwargs)
        else:
            raise ValueError(f"dos_method must be 'slope' or 'fit', got {dos_method!r}.")

        opt_labels = [l for l in op_dos.op_labels if _base_label(l) in ('mua', 'musp')]
        if not opt_labels:
            raise ValueError(
                f"The DOS fit produced no mua/musp on its op axis "
                f"(got {op_dos.op_labels}); nothing to hand to the DCS fit.")

        # --- stage 3: align each optical property onto the DCS grid -----
        aligned = {}       # label -> (n_dcs_time, 1, n_dcs_wl) values
        aligned_err = {}   # label -> same, or None
        for label in opt_labels:
            aligned[label] = _align_optical_da(
                op_dos.select(label), dcs_times, dcs_wls, time, wavelength).values
            if op_dos.uncertainty is not None:
                aligned_err[label] = _align_optical_da(
                    op_dos.uncertainty.sel(op=label),
                    dcs_times, dcs_wls, time, wavelength).values
            else:
                aligned_err[label] = None

        if any(bool(np.isnan(aligned[l]).any()) for l in opt_labels):
            import warnings
            warnings.warn(
                "Some DOS-derived mua/musp values are NaN (a DOS block whose "
                "fit did not converge); those DCS time slices will produce "
                "NaN bfi.", stacklevel=2)

        # --- stage 4: DCS fit -----------------------------------------
        fp = {'n': float(n)}
        for label in opt_labels:
            fp[label] = np.broadcast_to(
                aligned[label], (len(dcs_times), n_dcs_ch, len(dcs_wls)))
        fp.update(dcs_kwargs.pop('forward_parameters', {}) or {})

        op_bfi = dcs.fit_to_bfi(
            forward_parameters=fp, param_config=param_config,
            fixed_params=fixed_params, n_starts=n_starts, n_jobs=n_jobs,
            **dcs_kwargs)

        # --- augment the op axis with the mua/musp actually used -------
        op_da = op_bfi.data
        base = op_da.isel(op=0).drop_vars('op')          # (time, channel, wl) template
        tmpl_shape = base.shape
        u_base = (op_bfi.uncertainty.isel(op=0).drop_vars('op')
                  if op_bfi.uncertainty is not None else None)

        pieces = [op_da]
        err_pieces = [op_bfi.uncertainty] if u_base is not None else None
        for label in opt_labels:
            vals = np.broadcast_to(aligned[label], tmpl_shape)
            pieces.append(
                base.copy(data=vals).assign_coords(op=label).expand_dims('op'))
            if err_pieces is not None:
                e = aligned_err[label]
                edata = (np.broadcast_to(e, tmpl_shape) if e is not None
                         else np.full(tmpl_shape, np.nan))
                err_pieces.append(
                    u_base.copy(data=edata).assign_coords(op=label).expand_dims('op'))

        combined = xr.concat(pieces, dim='op').transpose(
            'time', 'channel', 'wavelength', 'op')
        combined.attrs = dict(op_da.attrs)
        combined_err = None
        if err_pieces is not None:
            combined_err = xr.concat(err_pieces, dim='op').transpose(
                'time', 'channel', 'wavelength', 'op')

        result = OptPropStream(
            data=combined,
            uncertainty=combined_err,
            obs_params=op_bfi.obs_params,
            obs_uncertainty=op_bfi.obs_uncertainty,
            covariance=op_bfi.covariance,
            forward_parameters=op_bfi.forward_parameters,
            probe=op_bfi.probe,
            name=output_name or f"{dos.name}_{dcs.name}_bfi",
            events=dcs.events,
            status='op',
            history=list(op_bfi.history),
        )
        result.add_history('fit_bfi_from_dos', {
            'dos_stream': dos.name, 'dcs_stream': dcs.name,
            'dos_window': window_info, 'dos_method': dos_method,
            'wavelength': wavelength, 'time': time, 'n': float(n),
            'opt_labels': list(opt_labels),
        })

        if store:
            op_dos.name = f"{dos.name}_op"
            self.add_stream(op_dos)
            self.add_stream(result)

        return result


# ---------------------------------------------------------------------------
# Helpers for Session.fit_bfi_from_dos -- the raw-DOS windowing and the
# DOS->DCS optical-property alignment. Kept at module level (pure array/xarray
# work, no Session state) so the method reads as the four pipeline stages.
# ---------------------------------------------------------------------------

def _fd_block_average(fd_stream, window, *, gap_factor, mismatch_factor, n_dcs):
    """Coherently average a raw FD stream to one frame per acquisition block."""
    import numpy as np
    import warnings

    t = np.asarray(fd_stream.data.time.values, dtype=float)
    n = len(t)

    if window is None:
        return fd_stream, {'mode': 'none', 'n_blocks': int(n)}

    if isinstance(window, str) and window == 'auto':
        if n < 3:
            return fd_stream, {'mode': 'auto->none', 'n_blocks': int(n),
                               'reason': 'fewer than 3 DOS frames'}
        dt = np.diff(t)
        typ = float(np.median(dt))
        gaps = np.where(dt > gap_factor * typ)[0]
        if len(gaps) == 0:
            ratio = max(n, n_dcs) / max(1, min(n, n_dcs))
            if ratio > mismatch_factor:
                warnings.warn(
                    f"DOS ({n} frames) and DCS ({n_dcs} frames) are both "
                    f"uniformly sampled but differ in length by {ratio:.1f}x, "
                    f"and no DOS acquisition blocks could be detected (no "
                    f"frame gap exceeds {gap_factor}x the {typ:.4g}s median "
                    f"sampling interval). This is an acquisition-design "
                    f"mismatch -- align the two streams yourself before "
                    f"calling fit_bfi_from_dos. Proceeding per DOS frame.",
                    stacklevel=3)
            return fd_stream, {'mode': 'auto->none', 'n_blocks': int(n),
                               'n_gaps': 0}
        bounds = [0, *(int(g) + 1 for g in gaps), n]
        segments = [np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
        return (_average_fd_segments(fd_stream, segments, t),
                {'mode': 'auto', 'n_blocks': len(segments),
                 'n_gaps': int(len(gaps)), 'gap_factor': gap_factor})

    if isinstance(window, str) and window == 'events':
        if fd_stream.events is None or len(fd_stream.events.table) == 0:
            raise ValueError("dos_window='events' but the DOS stream has no events.")
        onsets = np.sort(np.asarray(
            fd_stream.events.table['onset'].values, dtype=float))
        edges = np.concatenate(([t[0] - 1e-9], onsets, [t[-1] + 1e-9]))
        segments = _segments_from_edges(t, edges)
        return (_average_fd_segments(fd_stream, segments, t),
                {'mode': 'events', 'n_blocks': len(segments)})

    if isinstance(window, str):
        raise ValueError(
            f"dos_window={window!r} not understood. Use 'auto', None, "
            f"'events', a float (seconds), or a 1-D array of bin edges.")

    if np.ndim(window) == 0:                     # fixed boxcar width, seconds
        w = float(window)
        if w <= 0:
            raise ValueError(f"dos_window (seconds) must be positive, got {w}.")
        edges = np.arange(t[0], t[-1] + w, w)
        segments = _segments_from_edges(t, edges)
        return (_average_fd_segments(fd_stream, segments, t),
                {'mode': 'boxcar', 'window_s': w, 'n_blocks': len(segments)})

    edges = np.asarray(window, dtype=float)      # explicit bin edges
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("dos_window array must be 1-D with at least 2 edges.")
    segments = _segments_from_edges(t, edges)
    return (_average_fd_segments(fd_stream, segments, t),
            {'mode': 'edges', 'n_blocks': len(segments)})


def _segments_from_edges(t, edges):
    """Return frame-index arrays for each non-empty bin between consecutive edges."""
    import numpy as np
    idx = np.digitize(t, edges) - 1
    out = []
    for b in range(len(edges) - 1):
        members = np.where(idx == b)[0]
        if len(members):
            out.append(members)
    return out


def _average_fd_segments(fd_stream, segments, t):
    """Average FD segments onto a common time axis."""
    import numpy as np
    import xarray as xr

    data = fd_stream.data
    blocks = [data.isel(time=seg).mean('time', skipna=False) for seg in segments]
    centres = [float(np.mean(t[seg])) for seg in segments]
    new = xr.concat(blocks, dim='time').assign_coords(time=centres)
    new = new.transpose(*data.dims)
    new.attrs = dict(data.attrs)
    return fd_stream._rebuild(
        new, operation='fit_bfi_from_dos: block average',
        propagate_sidecars=False)


def _align_optical_da(da, target_time, target_wl, time_mode, wl_mode):
    """Align an optical DataArray onto a target time and wavelength grid."""
    import numpy as np
    import warnings

    if wl_mode not in ('interp', 'nearest'):
        raise ValueError(f"wavelength must be 'interp' or 'nearest', got {wl_mode!r}.")
    if time_mode not in ('interp', 'nearest'):
        raise ValueError(f"time must be 'interp' or 'nearest', got {time_mode!r}.")

    target_time = np.asarray(target_time, dtype=float)
    target_wl = np.asarray(target_wl, dtype=float)

    # --- wavelength: interp onto the DCS line(s), clamp outside the DOS band
    dos_wl = np.asarray(da.wavelength.values, dtype=float)
    near_wl = (da.sel(wavelength=target_wl, method='nearest')
               .assign_coords(wavelength=target_wl))
    if wl_mode == 'interp' and len(dos_wl) >= 2:
        w = da.interp(wavelength=target_wl).fillna(near_wl)
    else:
        if wl_mode == 'interp':
            warnings.warn(
                f"wavelength='interp' but the DOS fit has {len(dos_wl)} "
                f"wavelength(s); using 'nearest' for the wavelength handoff.",
                stacklevel=3)
        w = near_wl

    # --- time: interp onto the DCS timestamps, hold nearest edge outside cover
    dos_t = np.asarray(w.time.values, dtype=float)
    near_t = (w.sel(time=target_time, method='nearest')
              .assign_coords(time=target_time))
    if time_mode == 'interp' and len(dos_t) >= 2:
        a = w.interp(time=target_time).fillna(near_t)
    else:
        a = near_t

    return a.transpose('time', 'channel', 'wavelength')
