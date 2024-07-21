# EPEL
In this study, we first investigate the contribution of sequence-based features, including DNA shape, physicochemical properties and one-hot encodings based on encoded single nucleotide, and deep learningderived features from pre-trained chemical molecule language models based on BERT. Secondly, we propose EPEL, an effect predictor of synonymous mutations employing ensemble learning. Notably, the application of DNA shape features and deep learning-derived features based on chemical molecule represents a pioneer effect in assessing the impact of synonymous mutations. Compared to the exist state-of-the-art methods, EPEL demonstrates superior performance on independent test set. The details are summarized as follows. 

* data: in this paper, we used COSMIC as the training and independent test sets, respectively.

* out: it contains output result files, including prediction scores of test set.

* plot: it contains code and data for various chart referred to in the paper.

* src: it contains the code used in the project, including the processing of training and test data and modeling.

* model: models used and saved during the project. It contains data preprocessing, intermediate model and final ensemble leaning models. These are in three different folders:

  * data_processing_model

  * intermediate_model_all

  * clsUltimate_model_all

    

## Environment setup
We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/). See the file for detailed environment Settings at  `/environment/environment.yml`

* python 3.9.10
* scikit-learn 1.0.2
* xgboost  1.5.1
* catboost  1.2.2
* pymrmr  0.1.11

## Usage

Please see the template data at `/data`, it contains various characteristic and synonymous mutations in the form of variant call format. If you are trying to using EPEL with your own data, please process your data into the same format as it. Before using the model, you can read the help documentation.

```
python src/main.py -h

usage: main.py [-h] [-dataType DATATYPE] [-dbName DBNAME] [-dataPath DATAPATH] [-processingmodelPath PROCESSINGMODELPATH] [-intermodelPath INTERMODELPATH] [-clsmodelPath CLSMODELPATH] [-processedPath PROCESSEDPATH] [-interdataPath INTERDATAPATH] [-clsdataPath CLSDATAPATH] [-dimType DIMTYPE] [-r R]
```



```
python src/main.py 

    optional arguments:
      -h\, --help            show this help message and exit
      -dataType DATATYPE\    test or train
      -dbName DBNAME\        COSMIC
      -dataPath DATAPATH\    data path
      -processingmodelPath PROCESSINGMODELPATH\
                            processing model path
      -intermodelPath INTERMODELPATH\
                            intermediate model path
      -clsmodelPath CLSMODELPATH\
                            clsUltimate model path
      -processedPath PROCESSEDPATH\
                            processed data path
      -interdataPath INTERDATAPATH\
                            intermediate data path
      -clsdataPath CLSDATAPATH\
                            clsUltimate data path
      -dimType DIMTYPE\      dimensionType
      -r R\                  recurrenceLevel

```

## Examples

If you want to use these data to run the model, you can follow the example below to get started faster.

1. If you want to get the score in COSMIC via epSRel, you can ideally run like this:

   ```
   python src/main.py
   ```

2. If you want to retrain a new model, you can get it by training COSMIC data, of course, it will take a certain amount of time, please be patient.

   ```
   python src/main.py -dataType train -dbName COSMIC
   ```

## Reference

```
TO DO
```

