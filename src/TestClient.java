import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;

public class TestClient
{
    public static void main(String[] args) throws IOException
    {
        try (
                Socket echoSocket = new Socket("localhost", 4010);
                PrintWriter out =
                        new PrintWriter(echoSocket.getOutputStream(), true);
                BufferedReader in =
                        new BufferedReader(
                                new InputStreamReader(echoSocket.getInputStream()));
                BufferedReader stdIn =
                        new BufferedReader(
                                new InputStreamReader(System.in))
        )
        {
            String userInput;
            while ((userInput = stdIn.readLine()) != null)
            {
                out.println(userInput);
                System.out.println("Response: " + in.readLine());
            }
        }
        catch (UnknownHostException e)
        {
            System.err.println("Don't know about host localhost");
            System.exit(1);
        }
        catch (IOException e)
        {
            System.err.println("Couldn't get I/O for the connection to localhost");
            System.exit(1);
        }
    }
}
