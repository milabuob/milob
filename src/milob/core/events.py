import pandas as pd
import numpy as np
import warnings

class Events:
    def __init__(self, onsets=None, durations=None, values=None, labels=None):
        """
        Store experimental triggers and markers in a table.

        Parameters
        ----------
        onsets : array-like, optional
            Event onset times in seconds.
        durations : array-like, optional
            Event durations in seconds.
        values : array-like, optional
            Event amplitudes.
        labels : array-like, optional
            Condition label for each event.
        """
        data = {
            'onset': onsets if onsets is not None else [],
            'duration': durations if durations is not None else [],
            'value': values if values is not None else [],
            'label': labels if labels is not None else []
        }
        self._table = pd.DataFrame(data)
    
    
    # Property to access the table (read-only by default)
    @property
    def table(self):
        """Copy of the events table."""
        return self._table.copy()
    
    @table.setter
    def table(self, new_table):
        """Replace the events table."""
        if not isinstance(new_table, pd.DataFrame):
            raise TypeError("table must be a pandas DataFrame")
        
        # Validate columns
        required_cols = ['onset', 'duration', 'value', 'label']
        if not all(col in new_table.columns for col in required_cols):
            raise ValueError(f"Table must have columns: {required_cols}")
        
        self._table = new_table.copy()
    
    # Direct access methods
    def get_table(self, copy=True):
        """
        Return the events table.

        Parameters
        ----------
        copy : bool
            Return a copy rather than a reference to the underlying table.
            Default True.

        Returns
        -------
        pandas.DataFrame
            The events table.
        """
        return self._table.copy() if copy else self._table
    
    def set_table(self, table):
        """
        Replace the events table.

        Parameters
        ----------
        table : pandas.DataFrame
            New events table.
        """
        self.table = table
    
    # Add event
    def add_event(self, onset, duration, value, label):
        """
        Append a single event.

        Parameters
        ----------
        onset : float
            Onset time in seconds.
        duration : float
            Duration in seconds.
        value : float
            Event amplitude.
        label : str
            Condition label.
        """
        new_row = {'onset': onset, 'duration': duration, 'value': value, 'label': label}
        self._table = pd.concat([self._table, pd.DataFrame([new_row])], ignore_index=True)
        return self
    
    # Remove events
    def remove_event(self, index):
        """
        Remove one event.

        Parameters
        ----------
        index : int
            Row index of the event to remove.
        """
        self._table = self._table.drop(index).reset_index(drop=True)
        return self
    
    def remove_events_by_label(self, label):
        """
        Remove every event carrying a label.

        Parameters
        ----------
        label : str
            Condition label to remove.
        """
        self._table = self._table[self._table['label'] != label].reset_index(drop=True)
        return self
    
    # Update events
    def update_event(self, index, onset=None, duration=None, value=None, label=None):
        """
        Update fields of a single event.

        Parameters
        ----------
        index : int
            Row index of the event to update.
        onset, duration, value, label : optional
            Fields to replace. Omitted fields are left unchanged.
        """
        if onset is not None:
            self._table.at[index, 'onset'] = onset
        if duration is not None:
            self._table.at[index, 'duration'] = duration
        if value is not None:
            self._table.at[index, 'value'] = value
        if label is not None:
            self._table.at[index, 'label'] = label
        return self
    
    def update_events_by_label(self, old_label, new_label=None, duration=None, value=None):
        """
        Update every event carrying a label.

        Parameters
        ----------
        old_label : str
            Label identifying the events to update.
        new_label : str, optional
            Replacement label.
        duration, value : optional
            Fields to replace on the matched events.
        """
        mask = self._table['label'] == old_label
        if new_label is not None:
            self._table.loc[mask, 'label'] = new_label
        if duration is not None:
            self._table.loc[mask, 'duration'] = duration
        if value is not None:
            self._table.loc[mask, 'value'] = value
        return self
    
    
    @classmethod
    def from_bids(cls, file_path):
        """
        Read a BIDS _events.tsv or .csv file.

        Parameters
        ----------
        file_path : str
            Path to the events file.

        Returns
        -------
        Events
            Events parsed from the file.
        """
        sep = '\t' if file_path.endswith('.tsv') else ','
        df = pd.read_csv(file_path, sep=sep)

        # 1. Normalization mapping to match Milob Events standard.
        # 'trial_type' only maps to 'label' if the file doesn't already
        # define its own 'label' column (some datasets carry both a
        # generic BIDS trial_type and a more specific custom label).
        rename_map = {
            'OnsetTime_s': 'onset', 'onset': 'onset',
            'Duration': 'duration', 'duration': 'duration',
            'Condition': 'label',
        }
        if 'label' not in df.columns and 'trial_type' in df.columns:
            rename_map['trial_type'] = 'label'
        elif 'label' in df.columns and 'trial_type' in df.columns:
            warnings.warn(
                f"{file_path}: both 'trial_type' and 'label' columns present; "
                "using 'label' and discarding 'trial_type'."
            )
        df = df.rename(columns=rename_map)

        # 2. Handle missing columns. BIDS events.tsv has no standard 'value'
        # column, and 'duration'/'label' may simply be absent; NaN marks
        # these as genuinely unavailable rather than fabricating 0/1
        # placeholders that look like real trigger data downstream.
        for col in ('duration', 'value', 'label'):
            if col not in df.columns:
                df[col] = np.nan

        # 3. Clean up and sort
        core_cols = ['onset', 'duration', 'value', 'label']
        # Ensure only these columns exist and are in order
        df = df[core_cols].sort_values('onset').reset_index(drop=True)
        
        # 4. Use your existing from_dataframe to instantiate
        return cls.from_dataframe(df)
    
    
    # Query methods
    def get_events_by_label(self, label):
        """
        Return every event carrying a label.

        Parameters
        ----------
        label : str
            Condition label to match.

        Returns
        -------
        pandas.DataFrame
            The matching rows.
        """
        return self._table[self._table['label'] == label].copy()
    
    def get_event(self, index):
        """
        Return one event.

        Parameters
        ----------
        index : int
            Row index of the event.

        Returns
        -------
        dict
            The event's fields.
        """
        return self._table.iloc[index].to_dict()
    
    def filter_events(self, onset_min=None, onset_max=None, labels=None):
        """
        Select events by onset time and label.

        Parameters
        ----------
        onset_min : float, optional
            Earliest onset to keep.
        onset_max : float, optional
            Latest onset to keep.
        labels : str or list of str, optional
            Keep only events carrying these labels.

        Returns
        -------
        Events
            New Events object holding the selected events.
        """
        df = self._table.copy()
        
        if onset_min is not None:
            df = df[df['onset'] >= onset_min]
        if onset_max is not None:
            df = df[df['onset'] <= onset_max]
        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]
            df = df[df['label'].isin(labels)]
        
        return Events(
            onsets=df['onset'].values,
            durations=df['duration'].values,
            values=df['value'].values,
            labels=df['label'].values
        )
    
    # Sorting
    def sort_by_onset(self, inplace=True):
        """
        Sort events by onset time.

        Parameters
        ----------
        inplace : bool
            Modify this object in place. Default False.

        Returns
        -------
        Events
            Sorted events.
        """
        if inplace:
            self._table = self._table.sort_values('onset').reset_index(drop=True)
            return self
        else:
            sorted_table = self._table.sort_values('onset').reset_index(drop=True)
            return Events.from_dataframe(sorted_table)
    
    def remove_duplicates(self, subset=None, keep='first', inplace=True):
        """
        Remove duplicate events.

        Parameters
        ----------
        subset : str or list of str, optional
            Columns used to identify duplicates. Default ['onset', 'label'].
            Pass 'all' to require every column to match.
        keep : {'first', 'last', False}
            Which duplicate to retain. False drops all of them. Default 'first'.
        inplace : bool
            Modify this object in place. Default False.

        Returns
        -------
        Events
            Events with duplicates removed.

        Examples
        --------
        >>> events.remove_duplicates()
        >>> events.remove_duplicates(subset=['onset'])
        >>> events.remove_duplicates(subset='all')
        """
        if subset == 'all':
            subset = None  # pandas default: all columns
        elif subset is None:
            subset = ['onset', 'label']  # Default: same time + label
        
        if inplace:
            self._table = self._table.drop_duplicates(
                subset=subset, 
                keep=keep
            ).reset_index(drop=True)
            return self
        else:
            cleaned_table = self._table.drop_duplicates(
                subset=subset,
                keep=keep
            ).reset_index(drop=True)
            return Events.from_dataframe(cleaned_table)
    
    def merge_overlapping(self, tolerance=0.0, by_label=True, inplace=True):
        """
        Merge events that overlap in time.

        A merged event takes the earliest onset, spans to the latest offset, takes
        the mean of the values, and keeps the first label, or 'merged' when the
        labels differ.

        Parameters
        ----------
        tolerance : float
            Gap in seconds within which two events are still merged.
        by_label : bool
            Merge only events sharing a label. Default False.
        inplace : bool
            Modify this object in place. Default False.

        Returns
        -------
        Events
            Events with overlapping entries merged.
        """
        if self._table.empty:
            return self if inplace else Events()
        
        df = self._table.copy().sort_values('onset').reset_index(drop=True)
        #df['offset'] = df['onset'] + df['duration']
        
        merged_events = []
        
        if by_label:
            # Process each label separately
            for label in df['label'].unique():
                label_df = df[df['label'] == label].copy()
                merged_events.extend(
                    self._merge_group(label_df, tolerance, label)
                )
        else:
            # Merge all events together
            merged_events = self._merge_group(df, tolerance, None)
        
        merged_df = pd.DataFrame(merged_events)
        
        if inplace:
            self._table = merged_df
            return self
        else:
            return Events.from_dataframe(merged_df)
    
    def _merge_group(self, df, tolerance, default_label):
        """Merge one group of overlapping events into a single row."""
        if df.empty:
            return []
        
        merged = []
        current = df.iloc[0].to_dict()
        current_offset = current['onset'] + current['duration']
        values = [current['value']]
        labels = [current['label']]
        
        for _, row in df.iloc[1:].iterrows():
            # Check if overlapping or within tolerance
            if row['onset'] <= current_offset + tolerance:
                # Merge: extend current event
                current_offset = max(current_offset, row['onset'] + row['duration'])
                values.append(row['value'])
                labels.append(row['label'])
            else:
                # Save current and start new
                current['duration'] = current_offset - current['onset']
                current['value'] = np.mean(values)
                current['label'] = default_label if default_label else (
                    labels[0] if len(set(labels)) == 1 else 'merged'
                )
                merged.append(current)
                
                # Start new event
                current = row.to_dict()
                current_offset = current['onset'] + current['duration']
                values = [current['value']]
                labels = [current['label']]
        
        # Add last event
        current['duration'] = current_offset - current['onset']
        current['value'] = np.mean(values)
        current['label'] = default_label if default_label else (
            labels[0] if len(set(labels)) == 1 else 'merged'
        )
        merged.append(current)
        
        return merged
    

    
    # Properties for convenience
    @property
    def onsets(self):
        """Array of onset times in seconds."""
        return self._table['onset'].values
    
    @property
    def durations(self):
        """Array of durations in seconds."""
        return self._table['duration'].values
    
    @property
    def values(self):
        """Array of event amplitudes."""
        return self._table['value'].values
    
    @property
    def labels(self):
        """Array of condition labels."""
        return self._table['label'].values
    
    @property
    def conditions(self):
        """Unique condition labels."""
        return self._table['label'].unique()
    
    @property
    def n_events(self):
        """Total number of events."""
        return len(self._table)
    
    def n_events_by_condition(self):
        """
        Count events per condition.

        Returns
        -------
        dict of {str: int}
            Number of events carrying each label.
        """
        return self._table['label'].value_counts().to_dict()
    
    # Plotting
    def plot(self, **kwargs):
        """
        Plot these events against time, coloured by condition.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        from ..viz.events import plot_events
        return plot_events(self, **kwargs)

    # Display methods
    def __repr__(self):
        if self._table.empty:
            return "<Events | Empty>"
        conditions = self._table['label'].unique()
        return f"<Events | {len(self._table)} events | Conditions: {list(conditions)}>"
    
    def __str__(self):
        return self._table.to_string()
    
    def __len__(self):
        return len(self._table)
    
    # Index access
    def __getitem__(self, key):
        """Return an event by row index, or every event carrying a label."""
        if isinstance(key, str):
            # Column access
            return self._table[key].values
        else:
            # Row access
            return self._table.iloc[key].to_dict()
    
    def __setitem__(self, key, value):
        """Assign to a column of the events table."""
        if isinstance(key, str):
            self._table[key] = value
        else:
            raise TypeError("Can only set columns by name")
    
    # Iteration
    def __iter__(self):
        """Iterate over events as dicts."""
        for _, row in self._table.iterrows():
            yield row.to_dict()
    
    # Export methodshttps://www.readcube.com/library/348f9e0c-91ea-469f-be6a-a7f63e1749ab:d8e89b11-6a81-4f40-b2ca-805547f930ad
    def to_dict(self):
        """
        Export the events as a dictionary.

        Returns
        -------
        dict
            Column name to list of values.
        """
        return self._table.to_dict('list')
    
    def to_dataframe(self):
        """
        Export the events as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            Copy of the events table.
        """
        return self._table.copy()
    
    def to_csv(self, filepath):
        """
        Write the events to a CSV file.

        Parameters
        ----------
        filepath : str
            Destination path.
        """
        self._table.to_csv(filepath, index=False)
    
    @classmethod
    def from_dataframe(cls, df):
        """
        Build an Events object from a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Table with onset, duration, value and label columns.

        Returns
        -------
        Events
        """
        return cls(
            onsets=df['onset'].values,
            durations=df['duration'].values,
            values=df['value'].values,
            labels=df['label'].values
        )
    
    @classmethod
    def from_csv(cls, filepath):
        """
        Read events from a CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.

        Returns
        -------
        Events
        """
        df = pd.read_csv(filepath)
        return cls.from_dataframe(df)

    @classmethod
    def from_stim_matrix(cls, s_matrix, times, condition_labels=None):
        """
        Build events from a .nirs stimulus matrix.

        Parameters
        ----------
        s_matrix : np.ndarray
            Shape (n_time, n_conditions). Each column is a condition, and non-zero
            values mark event blocks.
        times : np.ndarray
            Shape (n_time,). Time vector in seconds matching the matrix rows.
        condition_labels : list of str, optional
            Name for each condition column. Defaults to 'cond_1', 'cond_2', ...

        Returns
        -------
        Events
        """
        s = np.atleast_2d(s_matrix)
        if s.shape[0] == 1:
            s = s.T  # ensure (n_time, n_conds)

        n_conds = s.shape[1]
        if condition_labels is None:
            condition_labels = [f"cond_{i + 1}" for i in range(n_conds)]

        onsets, durations, values, labels = [], [], [], []

        for col_idx in range(n_conds):
            col = s[:, col_idx]
            label = condition_labels[col_idx]
            i = 0
            while i < len(col):
                if col[i] != 0:
                    onset = times[i]
                    val = col[i]
                    j = i
                    while j < len(col) and col[j] != 0:
                        j += 1
                    duration = times[j - 1] - onset if j > i else 0.0
                    onsets.append(onset)
                    durations.append(duration)
                    values.append(float(val))
                    labels.append(label)
                    i = j
                else:
                    i += 1

        events = cls(onsets=onsets, durations=durations, values=values, labels=labels)
        events.sort_by_onset(inplace=True)
        return events