# Discrete units utilities

Single-file project to compute Phone purity, Cluster purity, PNMI, and to plot $P(\text{phone}|\text{code})$.

## Input files

The file containing the forced alignments must be a TSV file where the first column contains the audio files identifiers, and the second column contains the phones in order (separated by a "," by default).

The manifest file is a TSV where the first line is only the path to the dataset, and then the first columns contain the relative paths to the files and the second column contains the number of samples (not used here).

The units file is text file where the lines follow the file order of the manifest file, and each line contains the list of units (separated by a " " by default).

## Usage

See `example.ipynb`.

- Read files with `read_alignments` and `read_units`.
- With `count_matrix`, compute the 2D numpy array `count` of shape `(num_phones, codebook_size)` where `count[i, j]` contains the number of times a unit `j` has been associated to a phoneme `i`. This also returns the phonemes associated to the rows of the array (sorted by most present in the data).
- Compute Phone purity, Cluster purity and PNMI with `units_quality`.
- Compute the conditional probabilities $P(\text{phone}|\text{code})$ with `proba_phone_code(count)`.

> [!IMPORTANT]
> The function `count_matrix` also takes a `repeat` argument (by default "2"): each unit will be repeated this amount of times (with `np.repeat`). This is the ratio before the alignment frequency (eg 100Hz or one phone every 10ms) and the speech extraction frequency (eg 50Hz or one unit every 20ms).
