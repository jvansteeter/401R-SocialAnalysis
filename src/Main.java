import edu.stanford.nlp.simple.Sentence;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.ObjectInputStream;

public class Main
{
    public static void main(String[] args)
    {
//        SentimentAnalyzer test = new SentimentAnalyzer();
//        test.findSentiment("I hate everything so freaking much!");
//        MultilayerPerceptron perceptron = new MultilayerPerceptron();
//        perceptron.run();
//        ArffLoader loader = new ArffLoader();
//        loader.
        try
        {
//Reading training arff file
//            FileReader trainreader = new FileReader("processeddb.csv");
            CSVLoader loader = new CSVLoader();
            loader.setFile(new File("processeddb.csv"));
            Instances train = loader.getDataSet();
            train.remove(0);
//            Instances train = new Instances(trainreader);
            train.setClassIndex(train.numAttributes() - 1);

//Instance of NN
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            Dl4jMlpClassifier deepNet = (Dl4jMlpClassifier) (new ObjectInputStream(new FileInputStream("lia.model"))).readObject();
            deepNet.classifyInstance();

//Setting Parameters
            mlp.setLearningRate(0.1);
            mlp.setMomentum(0.2);
            mlp.setTrainingTime(5);
            mlp.setHiddenLayers("3");

            mlp.buildClassifier(train);

            loader.setFile(new File("test.csv"));
            Instance test2 = loader.getNextInstance(train);


            System.out.println(mlp.classifyInstance(test2));
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
}


