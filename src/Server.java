import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Server
{
    public static void main(String[] args) throws Exception
    {
        ExecutorService executor = Executors.newFixedThreadPool(5);

        try (ServerSocket serverSocket = new ServerSocket(4010))
        {
            System.out.println("Waiting for connections");
            while (true)
            {
                try
                {
                    Socket clientSocket = serverSocket.accept();
                    Runnable handler = new ConnectionHandler(clientSocket);
                    executor.execute(handler);
                }
                catch (IOException e)
                {
                    System.out.println("Exception caught when trying to listen on port 4980 or listening for a connection");
                    System.out.println(e.getMessage());
                }
            }
        }
    }
}
