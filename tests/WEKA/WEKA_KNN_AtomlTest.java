package weka.classifiers.lazy;

import static org.junit.Assert.*;
import org.junit.Test;
import org.junit.rules.TestName;
import org.junit.FixMethodOrder;
import org.junit.Rule;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;


import javax.annotation.Generated;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import smile.stat.hypothesis.KSTest;
import smile.stat.hypothesis.ChiSqTest;
import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.Instance;



/**
 * Automatically generated smoke and metamorphic tests.
 */
@Generated("atoml.testgen.TestclassGenerator")
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
@RunWith(Parameterized.class)
public class WEKA_KNN_AtomlTest {

    
    
    @Rule
    public TestName testname = new TestName();

    @Parameters(name = "{1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            { new String[]{"-K","3"}, "-K 3"}
           });
    }
    
    @Parameter
    public String[] parameters;
    
    @Parameter(1)
    public String parameterName;

    private void assertMorphTest(String evaluationType, String testcaseName, int iteration, int deviationsCounts, int deviationsScores, int testsize, int[] expectedMorphCounts, int[] morphedCounts, double[] expectedMorphedDistributions, double[] morphedDistributions, boolean passed, String errorMessage, String exception, String stacktrace) {
        if (passed) {
            if( "score_exact".equalsIgnoreCase(evaluationType) ) {
                String message = String.format("results different (deviations of scores: %d out of %d)", deviationsScores, testsize);
                assertTrue(message, deviationsScores==0);
            }
            else if( "class_exact".equalsIgnoreCase(evaluationType) ) {
                String message = String.format("results different (deviations of classes: %d out of %d)", deviationsCounts, testsize);
                assertTrue(message, deviationsCounts==0);
            }
            else if( "class_stat".equalsIgnoreCase(evaluationType) ) {
                double pValueCounts;
                if( deviationsCounts>0 ) {
                    pValueCounts = ChiSqTest.test(expectedMorphCounts, morphedCounts).pvalue;
                } else {
                    pValueCounts = 1.0;
                }
                String message = String.format("results significantly different, p-value = %f (deviations of classes: %d out of %d)", pValueCounts, deviationsCounts, testsize);
                assertTrue(message, pValueCounts>0.05);
            } 
            else if( "score_stat".equalsIgnoreCase(evaluationType) ) {
                double pValueKS;
                if( deviationsScores>0 ) {
                    pValueKS = KSTest.test(expectedMorphedDistributions, morphedDistributions).pvalue;
                    if (Double.isNaN(pValueKS)) {
                        pValueKS = 1.0;
                    }
                } else {
                    pValueKS = 1.0;
                }
                String message = String.format("score distributions significantly different, p-value = %f (deviations of scores: %d out of %d)", pValueKS, deviationsScores, testsize);
                assertTrue(message, pValueKS>0.05);
            } else {
                throw new RuntimeException("invalid evaluation type for morph test: " + evaluationType + " (allowed: exact, classification, score)");
            }
        } else {
            String message = errorMessage + '\n' + exception + '\n' + stacktrace;
            assertTrue(message, passed);
        }
    }
    
    private Instances loadData(String resourceName) {
        Instances data;
        InputStreamReader originalFile = new InputStreamReader(
                 this.getClass().getResourceAsStream(resourceName));
        try(BufferedReader reader = new BufferedReader(originalFile);) {
            data = new Instances(reader);
            reader.close();
        }
        catch (IOException e) {
            throw new RuntimeException(resourceName, e);
        }
        data.setClassIndex(data.numAttributes()-1);
        return data;
    }

    @Test(timeout=21600000)
    public void test_UniformSplit() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/UniformSplit_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/UniformSplit_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.lazy.IBk", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
            // get predictions on full testdata
            Evaluation eval = null;
            try {
                eval = new Evaluation(data);
                eval.evaluateModel(classifier, testdata);
            } catch (Exception ex) {
            }
            ArrayList pred = eval.predictions();
            // create a csv file
            String filePath = "predictions/pred_WEKA_KNN_UniformSplit_" + iter + ".csv";
            try {
                File outFile = new File(filePath);
                outFile.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriter = new FileWriter(filePath);
                // write header
                outWriter.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : pred) {
                    outWriter.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriter.close();
                System.out.println( "Predictions saved at: " + filePath );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
            // get predictions on full training data as test data
            Evaluation evalTrainingAsTest = null;
            try {
                evalTrainingAsTest = new Evaluation(data);
                evalTrainingAsTest.evaluateModel(classifier, data);
            } catch (Exception ex) {
            }
            ArrayList predTrainingAsTest = evalTrainingAsTest.predictions();
            // create a csv file
            String filePathTrainingAsTest = "predictions/pred_WEKA_KNN_UniformSplit_TrainingAsTest_" + iter + ".csv";
            try {
                File outFileTrainingAsTest = new File(filePathTrainingAsTest);
                outFileTrainingAsTest.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriterTrainingAsTest = new FileWriter(filePathTrainingAsTest);
                // write header
                outWriterTrainingAsTest.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : predTrainingAsTest) {
                    outWriterTrainingAsTest.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriterTrainingAsTest.close();
                System.out.println( "Predictions saved at: " + filePathTrainingAsTest );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
        }
    }

    @Test(timeout=21600000)
    public void test_RandomNumericSplit() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/RandomNumericSplit_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/RandomNumericSplit_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.lazy.IBk", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
            // get predictions on full testdata
            Evaluation eval = null;
            try {
                eval = new Evaluation(data);
                eval.evaluateModel(classifier, testdata);
            } catch (Exception ex) {
            }
            ArrayList pred = eval.predictions();
            // create a csv file
            String filePath = "predictions/pred_WEKA_KNN_RandomNumericSplit_" + iter + ".csv";
            try {
                File outFile = new File(filePath);
                outFile.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriter = new FileWriter(filePath);
                // write header
                outWriter.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : pred) {
                    outWriter.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriter.close();
                System.out.println( "Predictions saved at: " + filePath );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
            // get predictions on full training data as test data
            Evaluation evalTrainingAsTest = null;
            try {
                evalTrainingAsTest = new Evaluation(data);
                evalTrainingAsTest.evaluateModel(classifier, data);
            } catch (Exception ex) {
            }
            ArrayList predTrainingAsTest = evalTrainingAsTest.predictions();
            // create a csv file
            String filePathTrainingAsTest = "predictions/pred_WEKA_KNN_RandomNumericSplit_TrainingAsTest_" + iter + ".csv";
            try {
                File outFileTrainingAsTest = new File(filePathTrainingAsTest);
                outFileTrainingAsTest.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriterTrainingAsTest = new FileWriter(filePathTrainingAsTest);
                // write header
                outWriterTrainingAsTest.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : predTrainingAsTest) {
                    outWriterTrainingAsTest.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriterTrainingAsTest.close();
                System.out.println( "Predictions saved at: " + filePathTrainingAsTest );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
        }
    }

    @Test(timeout=21600000)
    public void test_BreastCancer() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/BreastCancer_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/BreastCancer_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.lazy.IBk", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
            // get predictions on full testdata
            Evaluation eval = null;
            try {
                eval = new Evaluation(data);
                eval.evaluateModel(classifier, testdata);
            } catch (Exception ex) {
            }
            ArrayList pred = eval.predictions();
            // create a csv file
            String filePath = "predictions/pred_WEKA_KNN_BreastCancer_" + iter + ".csv";
            try {
                File outFile = new File(filePath);
                outFile.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriter = new FileWriter(filePath);
                // write header
                outWriter.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : pred) {
                    outWriter.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriter.close();
                System.out.println( "Predictions saved at: " + filePath );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
            // get predictions on full training data as test data
            Evaluation evalTrainingAsTest = null;
            try {
                evalTrainingAsTest = new Evaluation(data);
                evalTrainingAsTest.evaluateModel(classifier, data);
            } catch (Exception ex) {
            }
            ArrayList predTrainingAsTest = evalTrainingAsTest.predictions();
            // create a csv file
            String filePathTrainingAsTest = "predictions/pred_WEKA_KNN_BreastCancer_TrainingAsTest_" + iter + ".csv";
            try {
                File outFileTrainingAsTest = new File(filePathTrainingAsTest);
                outFileTrainingAsTest.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriterTrainingAsTest = new FileWriter(filePathTrainingAsTest);
                // write header
                outWriterTrainingAsTest.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : predTrainingAsTest) {
                    outWriterTrainingAsTest.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriterTrainingAsTest.close();
                System.out.println( "Predictions saved at: " + filePathTrainingAsTest );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
        }
    }

    @Test(timeout=21600000)
    public void test_BreastCancerZNorm() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/BreastCancerZNorm_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/BreastCancerZNorm_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.lazy.IBk", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
            // get predictions on full testdata
            Evaluation eval = null;
            try {
                eval = new Evaluation(data);
                eval.evaluateModel(classifier, testdata);
            } catch (Exception ex) {
            }
            ArrayList pred = eval.predictions();
            // create a csv file
            String filePath = "predictions/pred_WEKA_KNN_BreastCancerZNorm_" + iter + ".csv";
            try {
                File outFile = new File(filePath);
                outFile.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriter = new FileWriter(filePath);
                // write header
                outWriter.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : pred) {
                    outWriter.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriter.close();
                System.out.println( "Predictions saved at: " + filePath );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
            // get predictions on full training data as test data
            Evaluation evalTrainingAsTest = null;
            try {
                evalTrainingAsTest = new Evaluation(data);
                evalTrainingAsTest.evaluateModel(classifier, data);
            } catch (Exception ex) {
            }
            ArrayList predTrainingAsTest = evalTrainingAsTest.predictions();
            // create a csv file
            String filePathTrainingAsTest = "predictions/pred_WEKA_KNN_BreastCancerZNorm_TrainingAsTest_" + iter + ".csv";
            try {
                File outFileTrainingAsTest = new File(filePathTrainingAsTest);
                outFileTrainingAsTest.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriterTrainingAsTest = new FileWriter(filePathTrainingAsTest);
                // write header
                outWriterTrainingAsTest.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : predTrainingAsTest) {
                    outWriterTrainingAsTest.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriterTrainingAsTest.close();
                System.out.println( "Predictions saved at: " + filePathTrainingAsTest );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
        }
    }

    @Test(timeout=21600000)
    public void test_BreastCancerMinMaxNorm() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/BreastCancerMinMaxNorm_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/BreastCancerMinMaxNorm_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.lazy.IBk", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
            // get predictions on full testdata
            Evaluation eval = null;
            try {
                eval = new Evaluation(data);
                eval.evaluateModel(classifier, testdata);
            } catch (Exception ex) {
            }
            ArrayList pred = eval.predictions();
            // create a csv file
            String filePath = "predictions/pred_WEKA_KNN_BreastCancerMinMaxNorm_" + iter + ".csv";
            try {
                File outFile = new File(filePath);
                outFile.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriter = new FileWriter(filePath);
                // write header
                outWriter.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : pred) {
                    outWriter.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriter.close();
                System.out.println( "Predictions saved at: " + filePath );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
            // get predictions on full training data as test data
            Evaluation evalTrainingAsTest = null;
            try {
                evalTrainingAsTest = new Evaluation(data);
                evalTrainingAsTest.evaluateModel(classifier, data);
            } catch (Exception ex) {
            }
            ArrayList predTrainingAsTest = evalTrainingAsTest.predictions();
            // create a csv file
            String filePathTrainingAsTest = "predictions/pred_WEKA_KNN_BreastCancerMinMaxNorm_TrainingAsTest_" + iter + ".csv";
            try {
                File outFileTrainingAsTest = new File(filePathTrainingAsTest);
                outFileTrainingAsTest.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriterTrainingAsTest = new FileWriter(filePathTrainingAsTest);
                // write header
                outWriterTrainingAsTest.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : predTrainingAsTest) {
                    outWriterTrainingAsTest.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriterTrainingAsTest.close();
                System.out.println( "Predictions saved at: " + filePathTrainingAsTest );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
        }
    }

    @Test(timeout=21600000)
    public void test_Wine() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/Wine_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/Wine_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.lazy.IBk", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
            // get predictions on full testdata
            Evaluation eval = null;
            try {
                eval = new Evaluation(data);
                eval.evaluateModel(classifier, testdata);
            } catch (Exception ex) {
            }
            ArrayList pred = eval.predictions();
            // create a csv file
            String filePath = "predictions/pred_WEKA_KNN_Wine_" + iter + ".csv";
            try {
                File outFile = new File(filePath);
                outFile.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriter = new FileWriter(filePath);
                // write header
                outWriter.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : pred) {
                    outWriter.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriter.close();
                System.out.println( "Predictions saved at: " + filePath );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
            // get predictions on full training data as test data
            Evaluation evalTrainingAsTest = null;
            try {
                evalTrainingAsTest = new Evaluation(data);
                evalTrainingAsTest.evaluateModel(classifier, data);
            } catch (Exception ex) {
            }
            ArrayList predTrainingAsTest = evalTrainingAsTest.predictions();
            // create a csv file
            String filePathTrainingAsTest = "predictions/pred_WEKA_KNN_Wine_TrainingAsTest_" + iter + ".csv";
            try {
                File outFileTrainingAsTest = new File(filePathTrainingAsTest);
                outFileTrainingAsTest.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriterTrainingAsTest = new FileWriter(filePathTrainingAsTest);
                // write header
                outWriterTrainingAsTest.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : predTrainingAsTest) {
                    outWriterTrainingAsTest.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriterTrainingAsTest.close();
                System.out.println( "Predictions saved at: " + filePathTrainingAsTest );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
        }
    }

    @Test(timeout=21600000)
    public void test_WineZNorm() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/WineZNorm_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/WineZNorm_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.lazy.IBk", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
            // get predictions on full testdata
            Evaluation eval = null;
            try {
                eval = new Evaluation(data);
                eval.evaluateModel(classifier, testdata);
            } catch (Exception ex) {
            }
            ArrayList pred = eval.predictions();
            // create a csv file
            String filePath = "predictions/pred_WEKA_KNN_WineZNorm_" + iter + ".csv";
            try {
                File outFile = new File(filePath);
                outFile.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriter = new FileWriter(filePath);
                // write header
                outWriter.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : pred) {
                    outWriter.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriter.close();
                System.out.println( "Predictions saved at: " + filePath );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
            // get predictions on full training data as test data
            Evaluation evalTrainingAsTest = null;
            try {
                evalTrainingAsTest = new Evaluation(data);
                evalTrainingAsTest.evaluateModel(classifier, data);
            } catch (Exception ex) {
            }
            ArrayList predTrainingAsTest = evalTrainingAsTest.predictions();
            // create a csv file
            String filePathTrainingAsTest = "predictions/pred_WEKA_KNN_WineZNorm_TrainingAsTest_" + iter + ".csv";
            try {
                File outFileTrainingAsTest = new File(filePathTrainingAsTest);
                outFileTrainingAsTest.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriterTrainingAsTest = new FileWriter(filePathTrainingAsTest);
                // write header
                outWriterTrainingAsTest.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : predTrainingAsTest) {
                    outWriterTrainingAsTest.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriterTrainingAsTest.close();
                System.out.println( "Predictions saved at: " + filePathTrainingAsTest );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
        }
    }

    @Test(timeout=21600000)
    public void test_WineMinMaxNorm() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/WineMinMaxNorm_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/WineMinMaxNorm_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.lazy.IBk", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
            // get predictions on full testdata
            Evaluation eval = null;
            try {
                eval = new Evaluation(data);
                eval.evaluateModel(classifier, testdata);
            } catch (Exception ex) {
            }
            ArrayList pred = eval.predictions();
            // create a csv file
            String filePath = "predictions/pred_WEKA_KNN_WineMinMaxNorm_" + iter + ".csv";
            try {
                File outFile = new File(filePath);
                outFile.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriter = new FileWriter(filePath);
                // write header
                outWriter.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : pred) {
                    outWriter.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriter.close();
                System.out.println( "Predictions saved at: " + filePath );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
            // get predictions on full training data as test data
            Evaluation evalTrainingAsTest = null;
            try {
                evalTrainingAsTest = new Evaluation(data);
                evalTrainingAsTest.evaluateModel(classifier, data);
            } catch (Exception ex) {
            }
            ArrayList predTrainingAsTest = evalTrainingAsTest.predictions();
            // create a csv file
            String filePathTrainingAsTest = "predictions/pred_WEKA_KNN_WineMinMaxNorm_TrainingAsTest_" + iter + ".csv";
            try {
                File outFileTrainingAsTest = new File(filePathTrainingAsTest);
                outFileTrainingAsTest.createNewFile();
            } catch (IOException e) {
                System.out.println( "Creating the csv file failed." );
                e.printStackTrace();
            }
            // write in csv
            try {
                FileWriter outWriterTrainingAsTest = new FileWriter(filePathTrainingAsTest);
                // write header
                outWriterTrainingAsTest.write("type,actual,prediction,weigth,prob_0,prob_1\n");
                // write predictions
                for (Object i : predTrainingAsTest) {
                    outWriterTrainingAsTest.write(i.toString().replace(" ", ",") + '\n');
                };
                outWriterTrainingAsTest.close();
                System.out.println( "Predictions saved at: " + filePathTrainingAsTest );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }
        }
    }


}