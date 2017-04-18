import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;

import java.io.*;
import java.net.*;

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
                if (prediction == 1.0)
                {
                    liveUpdate();
                }
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

    private void liveUpdate() throws IOException
    {
        System.out.println("Live update");
        String rawData = "update";
        String type = "application/x-www-form-urlencoded";
//        String encodedData = URLEncoder.encode(rawData, "UTF-8");
        URL url = new URL("http://localhost:3030/liveupdate");
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setDoOutput(true);
        connection.setRequestMethod("POST");
        connection.setRequestProperty("Content-Type", type);
        connection.setRequestProperty("charset", "utf-8");
        connection.setRequestProperty("Content-Length", Integer.toString(rawData.getBytes().length));
        OutputStream os = connection.getOutputStream();
        os.write(rawData.getBytes());
        os.flush();

        String response;
        BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));

        while ((response = reader.readLine()) != null)
        {
            System.out.println(response);
        }
        os.close();
        reader.close();
    }
}
