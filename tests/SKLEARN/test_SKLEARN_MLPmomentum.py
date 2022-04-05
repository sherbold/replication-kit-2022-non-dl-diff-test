import unittest
import xmlrunner
import pandas as pd
import numpy as np
import threading
import functools
import inspect
import math
import traceback
import warnings
import os
import time

from parameterized import parameterized
from pathlib import Path
from scipy.io.arff import loadarff
from scipy.stats import chisquare, ks_2samp
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier


class TestTimeoutException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# thanks to https://gist.github.com/vadimg/2902788
def timeout(duration, default=None):
    def decorator(func):
        class InterruptableThread(threading.Thread):
            def __init__(self, args, kwargs):
                threading.Thread.__init__(self)
                self.args = args
                self.kwargs = kwargs
                self.result = default
                self.daemon = True
                self.exception = None

            def run(self):
                try:
                    self.result = func(*self.args, **self.kwargs)
                except Exception as e:
                    self.exception = e

        @functools.wraps(func)
        def wrap(*args, **kwargs):
            it = InterruptableThread(args, kwargs)
            it.start()
            it.join(duration)
            if it.is_alive():
                raise TestTimeoutException('timeout after %i seconds for test %s' % (duration, func))
            if it.exception:
                raise it.exception
            return it.result
        return wrap
    return decorator

class test_SKLEARN_MLPmomentum(unittest.TestCase):

    params = [("{'alpha':0.0,'activation':'logistic','learning_rate_init':0.01,'max_iter':500,'hidden_layer_sizes':(50,50),'solver':'sgd','momentum':0.9,}", {'alpha':0.0,'activation':'logistic','learning_rate_init':0.01,'max_iter':500,'hidden_layer_sizes':(50,50),'solver':'sgd','momentum':0.9,}),
             ]

    def assert_morphtest(self, evaluation_type, testcase_name, iteration, deviations_class, deviations_score, pval_chisquare, pval_kstest, no_exception, exception_type, exception_message, exception_stacktrace):
        if no_exception:
            if evaluation_type=='score_exact':
                self.assertEqual(deviations_score, 0)
            elif evaluation_type=='class_exact':
                self.assertEqual(deviations_class, 0)
            elif evaluation_type=='score_stat':
                self.assertTrue(pval_kstest>0.05)
            elif evaluation_type=='class_stat':
                self.assertTrue(pval_chisquare>0.05)
            else:
                raise ValueError('invalid evaluation_type: %s (allowed: score_exact, class_exact, score_stat, class_stat' % evaluation_type)
        else:
            raise RuntimeError('%s encountered: %s %s' % exception_type, exception_message, exception_stacktrace)

    @parameterized.expand(params)
    @timeout(21600)
    def test_UniformSplit(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/UniformSplit_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/UniformSplit_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(testdata)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            predicted_label = classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            pred_prob = np.full((testdata_df.shape[0],2), -1)
            try:
                pred_prob = np.array(classifier.predict_proba(np.delete(testdata_df.values, classIndex, axis=1)))
            except AttributeError as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
                main_dir = Path(__file__).resolve().parents[2]
                prediction_file = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_UniformSplit_" + str(iter) + ".csv")
                pred_df = pd.DataFrame()
                pred_df['actual'] = testdata_df.classAtt
                pred_df['prediction'] = predicted_label
                pred_df['prob_0'] = pred_prob[:,0]
                pred_df['prob_1'] = pred_prob[:,1]
                pred_df.to_csv(prediction_file, header=True, index=False)
                print("Predictions saved at: " + prediction_file)
            except Exception as e:
                print("Saving the predictions of UniformSplit for SKLEARN_MLPmomentum failed: ", e)

            predicted_label_training_as_test = classifier.predict(np.delete(data_df.values, classIndex, axis=1))
            pred_prob_training_as_test = np.full((data_df.shape[0],2), -1)
            try:
                pred_prob_training_as_test = np.array(classifier.predict_proba(np.delete(data_df.values, classIndex, axis=1)))
            except Exception as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
              main_dir = Path(__file__).resolve().parents[2]
              prediction_file_training_as_test = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_UniformSplit_TrainingAsTest_" + str(iter) + ".csv")
              pred_df = pd.DataFrame()
              pred_df['actual'] = data_df.classAtt
              pred_df['prediction'] = predicted_label_training_as_test
              pred_df['prob_0'] = pred_prob_training_as_test[:,0]
              pred_df['prob_1'] = pred_prob_training_as_test[:,1]
              pred_df.to_csv(prediction_file_training_as_test, header=True, index=False)
              print("Predictions saved at: " + prediction_file_training_as_test)
            except Exception as e:
              print("Saving the predictions of UniformSplit with TrainingAsTest for SKLEARN_MLPmomentum failed: ", e)
    @parameterized.expand(params)
    @timeout(21600)
    def test_RandomNumericSplit(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/RandomNumericSplit_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/RandomNumericSplit_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(testdata)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            predicted_label = classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            pred_prob = np.full((testdata_df.shape[0],2), -1)
            try:
                pred_prob = np.array(classifier.predict_proba(np.delete(testdata_df.values, classIndex, axis=1)))
            except AttributeError as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
                main_dir = Path(__file__).resolve().parents[2]
                prediction_file = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_RandomNumericSplit_" + str(iter) + ".csv")
                pred_df = pd.DataFrame()
                pred_df['actual'] = testdata_df.classAtt
                pred_df['prediction'] = predicted_label
                pred_df['prob_0'] = pred_prob[:,0]
                pred_df['prob_1'] = pred_prob[:,1]
                pred_df.to_csv(prediction_file, header=True, index=False)
                print("Predictions saved at: " + prediction_file)
            except Exception as e:
                print("Saving the predictions of RandomNumericSplit for SKLEARN_MLPmomentum failed: ", e)

            predicted_label_training_as_test = classifier.predict(np.delete(data_df.values, classIndex, axis=1))
            pred_prob_training_as_test = np.full((data_df.shape[0],2), -1)
            try:
                pred_prob_training_as_test = np.array(classifier.predict_proba(np.delete(data_df.values, classIndex, axis=1)))
            except Exception as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
              main_dir = Path(__file__).resolve().parents[2]
              prediction_file_training_as_test = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_RandomNumericSplit_TrainingAsTest_" + str(iter) + ".csv")
              pred_df = pd.DataFrame()
              pred_df['actual'] = data_df.classAtt
              pred_df['prediction'] = predicted_label_training_as_test
              pred_df['prob_0'] = pred_prob_training_as_test[:,0]
              pred_df['prob_1'] = pred_prob_training_as_test[:,1]
              pred_df.to_csv(prediction_file_training_as_test, header=True, index=False)
              print("Predictions saved at: " + prediction_file_training_as_test)
            except Exception as e:
              print("Saving the predictions of RandomNumericSplit with TrainingAsTest for SKLEARN_MLPmomentum failed: ", e)
    @parameterized.expand(params)
    @timeout(21600)
    def test_BreastCancer(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/BreastCancer_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/BreastCancer_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(testdata)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            predicted_label = classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            pred_prob = np.full((testdata_df.shape[0],2), -1)
            try:
                pred_prob = np.array(classifier.predict_proba(np.delete(testdata_df.values, classIndex, axis=1)))
            except AttributeError as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
                main_dir = Path(__file__).resolve().parents[2]
                prediction_file = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_BreastCancer_" + str(iter) + ".csv")
                pred_df = pd.DataFrame()
                pred_df['actual'] = testdata_df.classAtt
                pred_df['prediction'] = predicted_label
                pred_df['prob_0'] = pred_prob[:,0]
                pred_df['prob_1'] = pred_prob[:,1]
                pred_df.to_csv(prediction_file, header=True, index=False)
                print("Predictions saved at: " + prediction_file)
            except Exception as e:
                print("Saving the predictions of BreastCancer for SKLEARN_MLPmomentum failed: ", e)

            predicted_label_training_as_test = classifier.predict(np.delete(data_df.values, classIndex, axis=1))
            pred_prob_training_as_test = np.full((data_df.shape[0],2), -1)
            try:
                pred_prob_training_as_test = np.array(classifier.predict_proba(np.delete(data_df.values, classIndex, axis=1)))
            except Exception as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
              main_dir = Path(__file__).resolve().parents[2]
              prediction_file_training_as_test = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_BreastCancer_TrainingAsTest_" + str(iter) + ".csv")
              pred_df = pd.DataFrame()
              pred_df['actual'] = data_df.classAtt
              pred_df['prediction'] = predicted_label_training_as_test
              pred_df['prob_0'] = pred_prob_training_as_test[:,0]
              pred_df['prob_1'] = pred_prob_training_as_test[:,1]
              pred_df.to_csv(prediction_file_training_as_test, header=True, index=False)
              print("Predictions saved at: " + prediction_file_training_as_test)
            except Exception as e:
              print("Saving the predictions of BreastCancer with TrainingAsTest for SKLEARN_MLPmomentum failed: ", e)
    @parameterized.expand(params)
    @timeout(21600)
    def test_BreastCancerZNorm(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/BreastCancerZNorm_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/BreastCancerZNorm_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(testdata)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            predicted_label = classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            pred_prob = np.full((testdata_df.shape[0],2), -1)
            try:
                pred_prob = np.array(classifier.predict_proba(np.delete(testdata_df.values, classIndex, axis=1)))
            except AttributeError as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
                main_dir = Path(__file__).resolve().parents[2]
                prediction_file = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_BreastCancerZNorm_" + str(iter) + ".csv")
                pred_df = pd.DataFrame()
                pred_df['actual'] = testdata_df.classAtt
                pred_df['prediction'] = predicted_label
                pred_df['prob_0'] = pred_prob[:,0]
                pred_df['prob_1'] = pred_prob[:,1]
                pred_df.to_csv(prediction_file, header=True, index=False)
                print("Predictions saved at: " + prediction_file)
            except Exception as e:
                print("Saving the predictions of BreastCancerZNorm for SKLEARN_MLPmomentum failed: ", e)

            predicted_label_training_as_test = classifier.predict(np.delete(data_df.values, classIndex, axis=1))
            pred_prob_training_as_test = np.full((data_df.shape[0],2), -1)
            try:
                pred_prob_training_as_test = np.array(classifier.predict_proba(np.delete(data_df.values, classIndex, axis=1)))
            except Exception as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
              main_dir = Path(__file__).resolve().parents[2]
              prediction_file_training_as_test = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_BreastCancerZNorm_TrainingAsTest_" + str(iter) + ".csv")
              pred_df = pd.DataFrame()
              pred_df['actual'] = data_df.classAtt
              pred_df['prediction'] = predicted_label_training_as_test
              pred_df['prob_0'] = pred_prob_training_as_test[:,0]
              pred_df['prob_1'] = pred_prob_training_as_test[:,1]
              pred_df.to_csv(prediction_file_training_as_test, header=True, index=False)
              print("Predictions saved at: " + prediction_file_training_as_test)
            except Exception as e:
              print("Saving the predictions of BreastCancerZNorm with TrainingAsTest for SKLEARN_MLPmomentum failed: ", e)
    @parameterized.expand(params)
    @timeout(21600)
    def test_BreastCancerMinMaxNorm(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/BreastCancerMinMaxNorm_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/BreastCancerMinMaxNorm_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(testdata)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            predicted_label = classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            pred_prob = np.full((testdata_df.shape[0],2), -1)
            try:
                pred_prob = np.array(classifier.predict_proba(np.delete(testdata_df.values, classIndex, axis=1)))
            except AttributeError as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
                main_dir = Path(__file__).resolve().parents[2]
                prediction_file = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_BreastCancerMinMaxNorm_" + str(iter) + ".csv")
                pred_df = pd.DataFrame()
                pred_df['actual'] = testdata_df.classAtt
                pred_df['prediction'] = predicted_label
                pred_df['prob_0'] = pred_prob[:,0]
                pred_df['prob_1'] = pred_prob[:,1]
                pred_df.to_csv(prediction_file, header=True, index=False)
                print("Predictions saved at: " + prediction_file)
            except Exception as e:
                print("Saving the predictions of BreastCancerMinMaxNorm for SKLEARN_MLPmomentum failed: ", e)

            predicted_label_training_as_test = classifier.predict(np.delete(data_df.values, classIndex, axis=1))
            pred_prob_training_as_test = np.full((data_df.shape[0],2), -1)
            try:
                pred_prob_training_as_test = np.array(classifier.predict_proba(np.delete(data_df.values, classIndex, axis=1)))
            except Exception as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
              main_dir = Path(__file__).resolve().parents[2]
              prediction_file_training_as_test = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_BreastCancerMinMaxNorm_TrainingAsTest_" + str(iter) + ".csv")
              pred_df = pd.DataFrame()
              pred_df['actual'] = data_df.classAtt
              pred_df['prediction'] = predicted_label_training_as_test
              pred_df['prob_0'] = pred_prob_training_as_test[:,0]
              pred_df['prob_1'] = pred_prob_training_as_test[:,1]
              pred_df.to_csv(prediction_file_training_as_test, header=True, index=False)
              print("Predictions saved at: " + prediction_file_training_as_test)
            except Exception as e:
              print("Saving the predictions of BreastCancerMinMaxNorm with TrainingAsTest for SKLEARN_MLPmomentum failed: ", e)
    @parameterized.expand(params)
    @timeout(21600)
    def test_Wine(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/Wine_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/Wine_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(testdata)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            predicted_label = classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            pred_prob = np.full((testdata_df.shape[0],2), -1)
            try:
                pred_prob = np.array(classifier.predict_proba(np.delete(testdata_df.values, classIndex, axis=1)))
            except AttributeError as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
                main_dir = Path(__file__).resolve().parents[2]
                prediction_file = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_Wine_" + str(iter) + ".csv")
                pred_df = pd.DataFrame()
                pred_df['actual'] = testdata_df.classAtt
                pred_df['prediction'] = predicted_label
                pred_df['prob_0'] = pred_prob[:,0]
                pred_df['prob_1'] = pred_prob[:,1]
                pred_df.to_csv(prediction_file, header=True, index=False)
                print("Predictions saved at: " + prediction_file)
            except Exception as e:
                print("Saving the predictions of Wine for SKLEARN_MLPmomentum failed: ", e)

            predicted_label_training_as_test = classifier.predict(np.delete(data_df.values, classIndex, axis=1))
            pred_prob_training_as_test = np.full((data_df.shape[0],2), -1)
            try:
                pred_prob_training_as_test = np.array(classifier.predict_proba(np.delete(data_df.values, classIndex, axis=1)))
            except Exception as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
              main_dir = Path(__file__).resolve().parents[2]
              prediction_file_training_as_test = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_Wine_TrainingAsTest_" + str(iter) + ".csv")
              pred_df = pd.DataFrame()
              pred_df['actual'] = data_df.classAtt
              pred_df['prediction'] = predicted_label_training_as_test
              pred_df['prob_0'] = pred_prob_training_as_test[:,0]
              pred_df['prob_1'] = pred_prob_training_as_test[:,1]
              pred_df.to_csv(prediction_file_training_as_test, header=True, index=False)
              print("Predictions saved at: " + prediction_file_training_as_test)
            except Exception as e:
              print("Saving the predictions of Wine with TrainingAsTest for SKLEARN_MLPmomentum failed: ", e)
    @parameterized.expand(params)
    @timeout(21600)
    def test_WineZNorm(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/WineZNorm_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/WineZNorm_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(testdata)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            predicted_label = classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            pred_prob = np.full((testdata_df.shape[0],2), -1)
            try:
                pred_prob = np.array(classifier.predict_proba(np.delete(testdata_df.values, classIndex, axis=1)))
            except AttributeError as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
                main_dir = Path(__file__).resolve().parents[2]
                prediction_file = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_WineZNorm_" + str(iter) + ".csv")
                pred_df = pd.DataFrame()
                pred_df['actual'] = testdata_df.classAtt
                pred_df['prediction'] = predicted_label
                pred_df['prob_0'] = pred_prob[:,0]
                pred_df['prob_1'] = pred_prob[:,1]
                pred_df.to_csv(prediction_file, header=True, index=False)
                print("Predictions saved at: " + prediction_file)
            except Exception as e:
                print("Saving the predictions of WineZNorm for SKLEARN_MLPmomentum failed: ", e)

            predicted_label_training_as_test = classifier.predict(np.delete(data_df.values, classIndex, axis=1))
            pred_prob_training_as_test = np.full((data_df.shape[0],2), -1)
            try:
                pred_prob_training_as_test = np.array(classifier.predict_proba(np.delete(data_df.values, classIndex, axis=1)))
            except Exception as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
              main_dir = Path(__file__).resolve().parents[2]
              prediction_file_training_as_test = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_WineZNorm_TrainingAsTest_" + str(iter) + ".csv")
              pred_df = pd.DataFrame()
              pred_df['actual'] = data_df.classAtt
              pred_df['prediction'] = predicted_label_training_as_test
              pred_df['prob_0'] = pred_prob_training_as_test[:,0]
              pred_df['prob_1'] = pred_prob_training_as_test[:,1]
              pred_df.to_csv(prediction_file_training_as_test, header=True, index=False)
              print("Predictions saved at: " + prediction_file_training_as_test)
            except Exception as e:
              print("Saving the predictions of WineZNorm with TrainingAsTest for SKLEARN_MLPmomentum failed: ", e)
    @parameterized.expand(params)
    @timeout(21600)
    def test_WineMinMaxNorm(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/WineMinMaxNorm_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/WineMinMaxNorm_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(testdata)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            predicted_label = classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            pred_prob = np.full((testdata_df.shape[0],2), -1)
            try:
                pred_prob = np.array(classifier.predict_proba(np.delete(testdata_df.values, classIndex, axis=1)))
            except AttributeError as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
                main_dir = Path(__file__).resolve().parents[2]
                prediction_file = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_WineMinMaxNorm_" + str(iter) + ".csv")
                pred_df = pd.DataFrame()
                pred_df['actual'] = testdata_df.classAtt
                pred_df['prediction'] = predicted_label
                pred_df['prob_0'] = pred_prob[:,0]
                pred_df['prob_1'] = pred_prob[:,1]
                pred_df.to_csv(prediction_file, header=True, index=False)
                print("Predictions saved at: " + prediction_file)
            except Exception as e:
                print("Saving the predictions of WineMinMaxNorm for SKLEARN_MLPmomentum failed: ", e)

            predicted_label_training_as_test = classifier.predict(np.delete(data_df.values, classIndex, axis=1))
            pred_prob_training_as_test = np.full((data_df.shape[0],2), -1)
            try:
                pred_prob_training_as_test = np.array(classifier.predict_proba(np.delete(data_df.values, classIndex, axis=1)))
            except Exception as e:
                print("The prediction of the probabilities failed. Values set to default (-1).")
            try:
              main_dir = Path(__file__).resolve().parents[2]
              prediction_file_training_as_test = os.path.join(main_dir, "predictions", "pred_SKLEARN_MLPmomentum_WineMinMaxNorm_TrainingAsTest_" + str(iter) + ".csv")
              pred_df = pd.DataFrame()
              pred_df['actual'] = data_df.classAtt
              pred_df['prediction'] = predicted_label_training_as_test
              pred_df['prob_0'] = pred_prob_training_as_test[:,0]
              pred_df['prob_1'] = pred_prob_training_as_test[:,1]
              pred_df.to_csv(prediction_file_training_as_test, header=True, index=False)
              print("Predictions saved at: " + prediction_file_training_as_test)
            except Exception as e:
              print("Saving the predictions of WineMinMaxNorm with TrainingAsTest for SKLEARN_MLPmomentum failed: ", e)


if __name__ == '__main__':
    unittest.main()
#    with open('results.xml', 'wb') as output:
#        unittest.main(
#            testRunner=xmlrunner.XMLTestRunner(output=output),
#            failfast=False, buffer=False, catchbreak=False)