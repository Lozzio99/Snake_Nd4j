package src.game.population;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import src.game.Display;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class Player
{
    MultiLayerNetwork model;
    double fitness;
    public Player()
    {
        this.loadModel();
    }
    public Player (MultiLayerNetwork model,double fit)
    {
        this.model = model;
        this.fitness = fit;
    }
    public double getFitness()
    {
        return this.fitness;
    }
    public Player(String directory, MultiLayerNetwork network)
    {
        this.model = network;
        new Display.SavingModel(new File(directory),true,network);
    }
    public static double fit;

    public MultiLayerNetwork getModel()
    {
        return this.model;
    }
    public static void writeFitness(double fit, int id)
    {
        try
        {
            File e = new File("models/0/fit"+id+".txt");
            FileWriter fw = new FileWriter(e);
            fw.write("fitness : "+ fit + " id : "+id+"\n");
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }
    public void loadModel()
    {
        try {
            this.model = ModelSerializer.restoreMultiLayerNetwork("trained_snake.zip");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
