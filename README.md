# QRS signals generation

### Get the mit data

For this purpose one can either use

```make download-mit```

or download the files manually from https://physionet.org/content/mitdb/1.0.0/#files 

The files should be stored in `data/` director creating the following structure:

```
...
├── data
│   ├── mit-bih-arrhythmia-database-1.0.0
│   │   ├── 100.atr
│   │   ├── 100.dat
│   │   ├── 100.hea
...
```

### QRS dataset

After loading mit data one can play with signals employing different transformations, time averaging, and clustering. Such options are provided in `qrs/` directory, where apart from data massaging one can export ready-to-use datasets using

```python -m qrs.export``` 


### Transformer

Datasets consisting of preprocessed QRS signals can be used for training purposes in the problem o QRS-like sequence generation. *Transformer* architecture (see `transformer/`) used here is based on the tutorial https://nlp.seas.harvard.edu/2018/04/03/attention.html 

### QRS generation

in preparation...
