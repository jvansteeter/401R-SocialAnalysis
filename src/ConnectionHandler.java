import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;

import java.io.*;
import java.net.Socket;

public class ConnectionHandler implements Runnable
{
    private Socket socket;

    public ConnectionHandler(Socket socket)
    {
        this.socket = socket;
    }

    @Override
    public void run()
    {
        try
        (
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        )
        {
            System.out.println("Socket Connected: " + socket.getInetAddress());
            FilteredClassifier svm = (FilteredClassifier) SerializationHelper.read("svm.model");
            ArffLoader arffLoader = new ArffLoader();
            arffLoader.setFile(new File("data/templateDataSet.arff"));
            Instances dataSet = arffLoader.getDataSet();
            dataSet.setClassIndex(1);
            Instance instance = dataSet.firstInstance();

            String tweet;
            while ((tweet = in.readLine()) != null)
            {
                instance.setValue(instance.attribute(0), tweet);
                double prediction = svm.classifyInstance(instance);
                System.out.println(socket.getInetAddress() + ": " + tweet);
                System.out.println("\tPrediction: " + (prediction == 0.0 ? "False" : "True"));
                out.println(prediction == 0.0 ? "False" : "True");
            }
        }
        catch (Exception e)
        {
            System.out.println("Exception caught when trying to listen on port 4010 or listening for a connection");
            System.out.println(e.getMessage());
            try
            {
                socket.close();
            }
            catch (IOException e1)
            {
                e1.printStackTrace();
            }
        }
    }
}
