package org.apache.spark.ml.classification;

import static org.junit.Assert.*;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.FixMethodOrder;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
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
import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instances;
import smile.stat.hypothesis.KSTest;
import smile.stat.hypothesis.ChiSqTest;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.OneHotEncoderModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.api.java.function.MapFunction;



/**
 * Automatically generated smoke and metamorphic tests.
 */
@Generated("atoml.testgen.TestclassGenerator")
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
@RunWith(Parameterized.class)
public class SPARK_MLP_AtomlAtomlTest {

    
    
    @Rule
    public TestName testname = new TestName();

    @Parameters(name = "{1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            { new String[]{"setMaxIter","500","setLayers","10, 50, 50, 2","setStepSize","0.01","setSolver","gd"}, "setMaxIter 500 setLayers 10, 50, 50, 2 setStepSize 0.01 setSolver gd"}
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
    

    private static SparkSession sparkSession;
    
    @BeforeClass
    public static void setUpClass() {
        sparkSession = SparkSession.builder().appName("Logistic_Default_AtomlTest").master("local[1]").getOrCreate();
        sparkSession.sparkContext().setLogLevel("WARN");
    }
    
    @AfterClass
    public static void tearDownClass() {
        sparkSession.stop();
    }

    private Dataset<Row> arffToDataset(String filename) {
        Instances data;
        InputStreamReader file = new InputStreamReader(this.getClass().getResourceAsStream(filename));
        try (BufferedReader reader = new BufferedReader(file);) {
            data = new Instances(reader);
            reader.close();
        } catch (IOException e) {
            throw new RuntimeException(filename, e);
        }

        List<StructField> fields = new LinkedList<>();
        for (int j = 0; j < data.numAttributes(); j++) {
            fields.add(DataTypes.createStructField(getNormalizedName(data.attribute(j)), DataTypes.DoubleType, false));
        }
        StructType schema = DataTypes.createStructType(fields);
        List<Row> rows = new LinkedList<>();
        for (int i = 0; i < data.size(); i++) {
            List<Double> valueList = new ArrayList<>(data.numAttributes());
            for (int j = 0; j < data.numAttributes(); j++) {
                valueList.add(data.instance(i).value(j));
            }

            rows.add(RowFactory.create(valueList.toArray()));
        }
        Dataset<Row> dataframe = sparkSession.createDataFrame(rows, schema);

        List<String> featureNames = new ArrayList<>();
        List<String> nominals = new LinkedList<>();
        List<String> nominalsOutput = new LinkedList<>();
        for (int j = 0; j < data.numAttributes() - 1; j++) {
            featureNames.add(getNormalizedName(data.attribute(j)));
            if (data.attribute(j).isNominal()) {
                nominals.add(getNormalizedName(data.attribute(j)));
                nominalsOutput.add(getNormalizedName(data.attribute(j)) + "_onehot");
            }
        }
        if (!nominals.isEmpty()) {
            OneHotEncoder oneHot = new OneHotEncoder().setInputCols(nominals.toArray(new String[0]))
                    .setOutputCols(nominalsOutput.toArray(new String[0])).setDropLast(false);
            OneHotEncoderModel oneHotModel = oneHot.fit(dataframe);
            dataframe = oneHotModel.transform(dataframe);
            dataframe = dataframe.drop(nominals.toArray(new String[0]));
            for (int j = nominals.size() - 1; j >= 0; j--) {
                dataframe = dataframe.withColumnRenamed(nominalsOutput.get(j), nominals.get(j));
            }
        }
        VectorAssembler va = new VectorAssembler().setInputCols(featureNames.toArray(new String[0]))
                .setOutputCol("features");
        dataframe = va.transform(dataframe);

        return dataframe;
    }
    
    private String getNormalizedName(Attribute attribute) {
        return attribute.name().replaceAll("\\.", "_");
    }
    
    private void setParameters(Object classifier, String[] parameters) {
        Method[] methods = classifier.getClass().getMethods();
        for( int i=0; i<parameters.length; i=i+2) {
            boolean methodFound = false;
            for(Method method : methods) {
                if( method.getName().equals(parameters[i])) {
                    methodFound = true;
                    for( java.lang.reflect.Parameter param : method.getParameters()) {
                        try {
                            if( "long".equals(param.getType().getName()) ) {
                                method.invoke(classifier, Long.parseLong(parameters[i+1]));
                                
                            }
                            else if( "int".equals(param.getType().getName()) ) {
                                method.invoke(classifier, Integer.parseInt(parameters[i+1]));
                            }
                            else if( "double".equals(param.getType().getName()) ) {
                                method.invoke(classifier, Double.parseDouble(parameters[i+1]));
                            }
                            else if( "boolean".equals(param.getType().getName()) ) {
                                method.invoke(classifier, Boolean.parseBoolean(parameters[i+1]));
                            }
                            else if( "java.lang.String".equals(param.getType().getName()) ) {
                                method.invoke(classifier, parameters[i+1]);
                            }
                            else if( "[I".equals(param.getType().getName()) ) {
                                String[] strNumbers = parameters[i+1].split(",");
                                int[] intArray = new int[strNumbers.length];
                                for( int j=0; j<strNumbers.length; j++) {
                                    intArray[j] = Integer.parseInt(strNumbers[j].trim());
                                }
                                method.invoke(classifier, intArray);
                            }
                            else {
                                throw new RuntimeException("Hyperparameter type not supported: " + param.getType().getName());
                            }
                        } catch (IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
                            throw new RuntimeException("Failure instantiating hyperparameter: " + parameters[i]);
                        }
                    }
                }
            }
            if( !methodFound ) {
                throw new RuntimeException("Invalid hyperparameters generated by atoml");
            }
        }
    }

    @Test(timeout=21600000)
    public void test_UniformSplit() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/UniformSplit_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/UniformSplit_" + iter + "_test.arff");

            MultilayerPerceptronClassifier classifier = new MultilayerPerceptronClassifier();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);

            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            Dataset<Row> pred = model.transform(testdata);
            Dataset<Row> predSelect = pred.select("classAtt", "prediction");
            try {
                predSelect = pred.select("classAtt", "prediction", "probability");
            } catch (Exception e) {
                predSelect = predSelect.withColumn("prob_0", functions.lit(-1));
                predSelect = predSelect.withColumn("prob_1", functions.lit(-1));
                System.out.print("The prediction of the probabilities failed. Values set to default (-1).");
            }
            // create a csv file
            String filePath = "predictions/pred_SPARK_MLP_UniformSplit_" + iter + ".csv";
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
                outWriter.write("actual,prediction,prob_0,prob_1\n");
                // write predictions
                List<String> predString = predSelect.map((MapFunction<Row, String>) row -> row.mkString(","), Encoders.STRING()).collectAsList();
                for (String s : predString) {
                    outWriter.write(s.replace("[", "").replace("]", "") + '\n');
                };
                outWriter.close();
                System.out.println( "Predictions saved at: " + filePath );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }

            // training data as test data
            Dataset<Row> predTrainingAsTest = model.transform(dataframe);
            Dataset<Row> predSelectTrainingAsTest = predTrainingAsTest.select("classAtt", "prediction");
            try {
                predSelectTrainingAsTest = predTrainingAsTest.select("classAtt", "prediction", "probability");
            } catch (Exception e) {
                predSelectTrainingAsTest = predSelectTrainingAsTest.withColumn("prob_0", functions.lit(-1));
                predSelectTrainingAsTest = predSelectTrainingAsTest.withColumn("prob_1", functions.lit(-1));
                System.out.print("The prediction of the probabilities failed. Values set to default (-1).");
            }
            // create a csv file
            String filePathTrainingAsTest = "predictions/pred_SPARK_MLP_UniformSplit_TrainingAsTest_" + iter + ".csv";
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
                outWriterTrainingAsTest.write("actual,prediction,prob_0,prob_1\n");
                // write predictions
                List<String> predStringTrainingAsTest = predSelectTrainingAsTest.map((MapFunction<Row, String>) row -> row.mkString(","), Encoders.STRING()).collectAsList();
                for (String s : predStringTrainingAsTest) {
                    outWriterTrainingAsTest.write(s.replace("[", "").replace("]", "") + '\n');
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
        	Dataset<Row> dataframe = arffToDataset("/smokedata/RandomNumericSplit_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/RandomNumericSplit_" + iter + "_test.arff");

            MultilayerPerceptronClassifier classifier = new MultilayerPerceptronClassifier();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);

            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            Dataset<Row> pred = model.transform(testdata);
            Dataset<Row> predSelect = pred.select("classAtt", "prediction");
            try {
                predSelect = pred.select("classAtt", "prediction", "probability");
            } catch (Exception e) {
                predSelect = predSelect.withColumn("prob_0", functions.lit(-1));
                predSelect = predSelect.withColumn("prob_1", functions.lit(-1));
                System.out.print("The prediction of the probabilities failed. Values set to default (-1).");
            }
            // create a csv file
            String filePath = "predictions/pred_SPARK_MLP_RandomNumericSplit_" + iter + ".csv";
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
                outWriter.write("actual,prediction,prob_0,prob_1\n");
                // write predictions
                List<String> predString = predSelect.map((MapFunction<Row, String>) row -> row.mkString(","), Encoders.STRING()).collectAsList();
                for (String s : predString) {
                    outWriter.write(s.replace("[", "").replace("]", "") + '\n');
                };
                outWriter.close();
                System.out.println( "Predictions saved at: " + filePath );
            } catch (IOException e) {
                System.out.println( "Writing the predictions to csv file failed." );
                e.printStackTrace();
            }

            // training data as test data
            Dataset<Row> predTrainingAsTest = model.transform(dataframe);
            Dataset<Row> predSelectTrainingAsTest = predTrainingAsTest.select("classAtt", "prediction");
            try {
                predSelectTrainingAsTest = predTrainingAsTest.select("classAtt", "prediction", "probability");
            } catch (Exception e) {
                predSelectTrainingAsTest = predSelectTrainingAsTest.withColumn("prob_0", functions.lit(-1));
                predSelectTrainingAsTest = predSelectTrainingAsTest.withColumn("prob_1", functions.lit(-1));
                System.out.print("The prediction of the probabilities failed. Values set to default (-1).");
            }
            // create a csv file
            String filePathTrainingAsTest = "predictions/pred_SPARK_MLP_RandomNumericSplit_TrainingAsTest_" + iter + ".csv";
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
                outWriterTrainingAsTest.write("actual,prediction,prob_0,prob_1\n");
                // write predictions
                List<String> predStringTrainingAsTest = predSelectTrainingAsTest.map((MapFunction<Row, String>) row -> row.mkString(","), Encoders.STRING()).collectAsList();
                for (String s : predStringTrainingAsTest) {
                    outWriterTrainingAsTest.write(s.replace("[", "").replace("]", "") + '\n');
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