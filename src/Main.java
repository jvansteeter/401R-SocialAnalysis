import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.TweetToSparseFeatureVector;

import java.io.File;

public class Main
{
    public static void main(String[] args)
    {
        try
        {
//Reading training arff file
//            FileReader trainreader = new FileReader("processeddb.csv");
//            CSVLoader loader = new CSVLoader();
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File("data/deepData.arff"));
            Instances dataSet = loader.getDataSet();
//            System.out.println(test);
//            train.remove(0);
////            Instances train = new Instances(trainreader);
//            train.setClassIndex(train.numAttributes() - 1);

//Instance of NN
            /*TweetToEmbeddingsFeatureVector filter = new TweetToEmbeddingsFeatureVector();
            filter.setEmbeddingFileName("resources/w2v.twitter.edinburgh10M.400d.csv.gz");
            filter.setOptions(new String[]{"-I", "1", "-S", "2", "-K", "15", "-L", "-O"});
            filter.setInputFormat(test);
//            filter.input(test.firstInstance());
//            filter.useFilter()
            Instances dataSet = Filter.useFilter(test, filter);

            Reorder reorder = new Reorder();
            reorder.setOptions(new String[]{"-R", "3-last,2"});
            reorder.setInputFormat(dataSet);
            dataSet = Filter.useFilter(dataSet, reorder);*/


            /*TweetToSparseFeatureVector filter = new TweetToSparseFeatureVector();
            filter.setCalculateCharNgram(true);
            filter.setCharNgramMaxDim(5);
            filter.setCharNgramMinDim(3);
            filter.setCleanTokens(true);
            filter.setClustNgramMaxDim(1);
            filter.setDebug(false);
            filter.setDoNotCheckCapabilities(false);
            filter.setFreqWeights(true);
            filter.setMinAttDocs(2);
            filter.setNegateTokens(true);
            filter.setPosNgramMaxDim(1);
            filter.setTextIndex(1);
            filter.setToLowerCase(true);
            filter.setWordNgramMaxDim(2);
            filter.setInputFormat(instances);

            Instances dataSet = Filter.useFilter(instances, filter);

//            System.out.println(instances);
            System.out.println(dataSet);
            for (int i = 0; i < dataSet.size(); i++)
            {
                System.out.println(dataSet.get(i).stringValue(0) + " " + dataSet.get(i).value(1));
            }
            System.out.println("OK");*/

//            Dl4jMlpClassifier lia = new Dl4jMlpClassifier();
//            lia.setOptions(new String[]{"-S", "1",
//                    //"-iterator", "weka.dl4j.iterators.ConvolutionalInstancesIterator -height 1 -numChannels 1 -bs 256 -width 6000",
//                    //"-layers", "weka.dl4j.layers.ConvolutionLayer -nFilters 100 -activation identity -adamMeanDecay 0.9 -adamVarDecay 0.999 -biasInit 1.0 -biasL1 0.0 -biasL2 0.0 -blr 0.01 -mode Truncate -cudnnAlgoMode PREFER_FASTEST -dist \"weka.dl4j.distribution.NormalDistribution -mean 0.001 -std 1.0\" -dropout 0.0 -epsilon 1.0E-6 -gradientNormalization None -gradNormThreshold 1.0 -kernelSizeX 300 -kernelSizeY 1 -L1 0.0 -L2 0.0 -name \"Convolution layer\" -lr 0.01 -momentum 0.9 -momentumSchedule {} -paddingX 0 -paddingY 0 -rho 0.0 -rmsDecay 0.95 -strideX 100 -strideY 1 -updater NESTEROVS -weightInit XAVIER",
//                    //"-layers", "weka.dl4j.layers.OutputLayer -activation softmax -adamMeanDecay 0.9 -adamVarDecay 0.999 -biasInit 1.0 -biasL1 0.0 -biasL2 0.0 -blr 0.01 -dist \"weka.dl4j.distribution.NormalDistribution -mean 0.001 -std 1.0\" -dropout 0.0 -epsilon 1.0E-6 -gradientNormalization None -gradNormThreshold 1.0 -L1 0.0 -L2 0.0 -name \"Output layer\" -lr 0.01 -lossFn LossMCXENT() -momentum 0.9 -momentumSchedule {} -rho 0.0 -rmsDecay 0.95 -updater NESTEROVS -weightInit XAVIER",
//                    "-logFile", "weka.log", "-numEpochs", "200", "-algorithm", "STOCHASTIC_GRADIENT_DESCENT"});
//            ConvolutionalInstancesIterator iterator = new ConvolutionalInstancesIterator();
//            iterator.setOptions(new String[]{"-height", "1", "-numChannels", "1", "-bs", "256", "-width", "6000"});
//            lia.setDataSetIterator(iterator);
//            ConvolutionLayer layer1 = new ConvolutionLayer();
//            layer1.setOptions(new String[]{"-nFilters", "100", "-activation", "identity", "-adamMeanDecay", "0.9", "-adamVarDecamvny", "0.999", "-biasInit", "1.0", "-biasL1", "0.0",
//                    "-biasL2", "0.0", "-blr", "0.01", "-mode", "Truncate", "-cudnnAlgoMode", "PREFER_FASTEST",
//                    "-dropout", "0.0", "-epsilon", "1.0E-6", "-gradientNormalization", "None", "-gradNormThreshold", "1.0", "-kernelSizeX", "300", "-kernelSizeY", "1", "-L1", "0.0", "-L2", "0.0",
//                    "-name", "Convolution layer", "-lr", "0.01", "-momentum", "0.9", "-momentumSchedule", "{}", "-paddingX", "0", "-paddingY", "0", "-rho", "0.0", "-rmsDecay", "0.95", "-strideX", "100",
//                    "-strideY", "1", "-updater", "NESTEROVS", "-weightInit", "XAVIER"});
//            NormalDistribution normalDistribution = new NormalDistribution();
//            normalDistribution.setOptions(new String[]{"-mean", "0.001", "-std", "1.0"});
//            layer1.setDist(normalDistribution);
//            OutputLayer layer2 = new OutputLayer();
//            layer2.setOptions(new String[]{"-activation", "softmax", "-adamMeanDecay", "0.9", "-adamVarDecay", "0.999", "-biasInit", "1.0", "-biasL1", "0.0", "-biasL2", "0.0", "-blr", "0.01",
//                    //-dist "weka.dl4j.distribution.NormalDistribution -mean 0.001 -std 1.0",
//                    "-dropout", "0.0", "-epsilon", "1.0E-6", "-gradientNormalization", "None", "-gradNormThreshold", "1.0", "-L1", "0.0", "-L2", "0.0", "-name", "Output layer", "-lr", "0.01", "-lossFn", "LossMCXENT()",
//                    "-momentum", "0.9", "-momentumSchedule", "{}", "-rho", "0.0", "-rmsDecay", "0.95", "-updater", "NESTEROVS", "-weightInit", "XAVIER"});
//            NormalDistribution normalDistribution1 = new NormalDistribution();
//            normalDistribution1.setOptions(new String[]{"-mean", "0.001", "-std", "1.0"});
//            layer2.setDist(normalDistribution1);
//            lia.setLayers(new Layer[]{layer1, layer2});

            FilteredClassifier svm = (FilteredClassifier) SerializationHelper.read("svm.model");
            dataSet.setClassIndex(1);

            int correct = 0;
            int incorrect = 0;
            int threats = 0;
            int correctThreats = 0;
            for (int i = 0; i < dataSet.size(); i++)
            {
                Instance instance = dataSet.get(i);
                double result = svm.classifyInstance(instance);
                double target = instance.value(1);
                if (result == target)
                {
                    correct++;
                    System.out.print("Correct: ");
                    if (target == 1)
                    {
                        correctThreats++;
                    }
                }
                else
                {
                    incorrect++;
                    System.out.print("Incorrect: ");
                }
                if (target == 1)
                {
                    threats++;
                }
                System.out.println(result);
            }
            System.out.println("Correct: " + correct);
            System.out.println("Incorrect: " + incorrect);
            System.out.println("Threats: " + threats);
            System.out.println("%: " + (double)correct/(double)(incorrect + correct));
            System.out.println("%: " + (double)correctThreats/(double)(threats));
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
}


