package src.game;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.io.File;

/**
 * Runs a game of Snake.
 * Uses the arrow keys to move the Snake.
 * Click F1, F2, F3, F4 or F5 to change the color.
 */
public class Display extends JFrame {

    private Engine engine;
    File locationToSave = new File("trained_snake.zip");

    private Display() {
        engine = createEngine();
        setWindowProperties();
    }

    private Engine createEngine () {

        Container cp = getContentPane();
        System system = new System();
        Engine engine = new Engine(system);

        int canvasWidth = Configuration.SQUARE_SIZE * Configuration.BOARD_COLUMNS;
        int canvasHeight = Configuration.SQUARE_SIZE * Configuration.BOARD_ROWS;
        engine.setPreferredSize(new Dimension(canvasWidth, canvasHeight));

        addKeyListener(new MyKeyAdapter());

        cp.add(engine);

        return engine;
    }

    private void setWindowProperties () {
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setTitle("Snake - Score: 0");
        setResizable(false);
        pack();
        setVisible(true);
        setLocationRelativeTo(null);// Center window
    }

    private void startGame (Engine engine) {
        Thread th = new Thread(engine);
        th.start();
    }

    /**
     * Contains the game loop.
     */
    private class Engine extends JPanel implements Runnable {

        private System system;
        private boolean running = false;

        private Engine(System system) {
            this.system = system;
        }

        @Override
        protected void paintComponent(Graphics graphics) {
            super.paintComponent(graphics);

            // Ensures that it will run smoothly on Linux.
            if (java.lang.System.getProperty("os.name").equals("Linux")) {
                Toolkit.getDefaultToolkit().sync();
            }

            setBackground(Configuration.backgroundColor);
            system.paint(graphics);
        }

        public void run () {

            long lastTime = java.lang.System.nanoTime();
            double elapsedTime = 0.0;
            double FPS = 10.0;

            // Game loop.
            while (true) {

                final long now = java.lang.System.nanoTime();
                elapsedTime += ((now - lastTime) / 1_000_000_000d) * FPS;
                lastTime = java.lang.System.nanoTime();

                if (elapsedTime >= 1) {
                    system.update();
                    //System.out.println(gameBoard.toString());
                    setTitle("Score: " + system.getScore()+ " f- "+ System.bestFitness+ " p- "+ System.goodPlayers);
                    if (java.lang.System.nanoTime()-now>= 1E10)
                        java.lang.System.out.println();
                    elapsedTime--;
                }

                sleep();
                
                //7/28/2017
                //If the rainbow theme is selected lets update the color
                if (Configuration.getTheme() == Configuration.Theme.Rainbow)
                    Configuration.changeColor();
                
                repaint();
            }
        }

    }

    /**
     * Sleep for 10 milliseconds.
     */
    private void sleep () {
        try {
            Thread.sleep(10);
        } catch (InterruptedException ex) {
            ex.printStackTrace();
        }
    }

    private class MyKeyAdapter extends KeyAdapter {

        @Override
        public void keyPressed(KeyEvent keyEvent)
        {

            if (!engine.running && keyEvent.getKeyCode() != KeyEvent.VK_F1 && keyEvent.getKeyCode() != KeyEvent.VK_F2 && keyEvent.getKeyCode() != KeyEvent.VK_F3 && keyEvent.getKeyCode() != KeyEvent.VK_F4 && keyEvent.getKeyCode() != KeyEvent.VK_F5) {
                startGame(engine);
                engine.running = true;
            }

            if (keyEvent.getKeyCode()==KeyEvent.VK_SPACE)
            {
                new SavingModel(locationToSave, true, System.brain);
            }
            if (keyEvent.getKeyCode() ==KeyEvent.VK_7)
            {
                engine.system.reset();
                System.mutate();
            }
            if (keyEvent.getKeyCode() ==KeyEvent.VK_8)
            {
                engine.system.load();
            }
            if (keyEvent.getKeyCode() ==KeyEvent.VK_6)
            {
                engine.system.rs();
            }

            /*
            if (keyEvent.getKeyCode() == KeyEvent.VK_LEFT) {
                engine.gameBoard.directionLeft();
            } else if (keyEvent.getKeyCode() == KeyEvent.VK_RIGHT) {
                engine.gameBoard.directionRight();
            } else if (keyEvent.getKeyCode() == KeyEvent.VK_UP) {
                engine.gameBoard.directionUp();
            } else if (keyEvent.getKeyCode() == KeyEvent.VK_DOWN) {
                engine.gameBoard.directionDown();
            }
            */

            if (keyEvent.getKeyCode() == KeyEvent.VK_F1) {
                Configuration.Dark();
                repaint();
            }
            else if (keyEvent.getKeyCode() == KeyEvent.VK_F2) {
                Configuration.Sky();
                repaint();
            }
            else if (keyEvent.getKeyCode() == KeyEvent.VK_F3) {
                Configuration.Mud();
                repaint();
            }
            else if (keyEvent.getKeyCode() == KeyEvent.VK_F4) {
                Configuration.Sand();
                repaint();
            }
            else if (keyEvent.getKeyCode() == KeyEvent.VK_F5) {
                Configuration.Rainbow();
                repaint();
            }
        }

    }

    public static class SavingModel
    {
        public SavingModel(File file, boolean updateSaver, MultiLayerNetwork model)
        {
            try
            {
                ModelSerializer.writeModel(model,file,updateSaver);

            }catch (Exception e )
            {
                e.printStackTrace();
            }
        }
    }
    public static void main(String[] args) {
        SwingUtilities.invokeLater(Display::new);
    }
}