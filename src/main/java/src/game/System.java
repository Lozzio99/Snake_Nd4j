package src.game;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import src.game.population.Player;

import java.awt.*;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Represents the environment where the Snake moves a food spawns.
 * <br/>
 * There are some special rules as to how the Snake can move. If the Snake's size
 * is 1, it can move in any direction. If the Snake's size is greater than 1, it
 * cannot move 180 degrees. Example: if the Snake is moving right, it cannot
 * immediately change its direction to left because it would run into itself.
 */
class System {

    private Square food;
    private Snake snake;
    private int score = 0;
    public static MultiLayerNetwork brain;
    private double [] visualfield ;
    private int timeOver = 500;
    private double fitness;
    public static int goodPlayers = 0;
    private static int nLastGood = 0;
    private static int nLastRestore = 0;
    public static double bestFitness = 0;
    private final static int maxP = 5;
    private static MultiLayerNetwork best;
    private static Player[] population = new Player[maxP];


    //private VisualNetwork visualNetwork = new VisualNetwork();

    /**
     * Keep track of the last move so that the Snake cannot do 180 degree turns,
     * only 90 degree turns.
     */
    private Direction movement = Direction.DOWN;
    private Direction lastMove = movement;

    /**
     * Constructs the board.
     */
    System()
    {
        this.snake = new Snake();
        /*
        TODO: Probably search for a better configuration, this is is kinda stuck since the beginning
         */
        brain = new Player().getModel();
        brain.init();
        newFood();
        update();
    }

    /**
     * Move the Snake.
     */
    void update ()
    {
        setField();
        double [] prediction = brain.output(Nd4j.create(this.visualfield,1, Configuration.BOARD_COLUMNS* Configuration.BOARD_ROWS)).toDoubleVector();
        double max = 0;
        int index = -1;
        for (int i =0; i< prediction.length; i++)
        {
            if (prediction[i]>max)
            {
                max = prediction[i];
                index = i;
            }
        }
        switch (index) {
            case 0 -> directionDown();
            case 1 -> directionUp();
            case 2 -> directionLeft();
            case 3 -> directionRight();
        }
        moveSnake();
        updateFitness();

    }

    private void updateFitness()
    {
        int steps = 500-timeOver;
        fitness = f(steps,getScore());

        if (timeOver>0)
        {
            timeOver--;
            if (fitness>bestFitness)
            {
                bestFitness = fitness;
                best = brain;
            }
        }
        else reset();
    }
    void load()
    {
        brain = new Player().getModel();
        this.snake = new Snake();
        this.score = 0;
        this.fitness = 0;
        timeOver = 500;
        newFood();
        update();
    }

    private void sort()
    {
        for (int i = 0; i< maxP; i++)
        {
            if(fitness>population[i].getFitness())
            {
                population[i] = new Player(brain,fitness);
                break;
            }
        }
    }

    String stats()
    {
        String s = "Moves left : "+ timeOver + " fitness : "+ fitness;
        s+= "best fitness "+ bestFitness + " \npopulation size "+ goodPlayers;
        s+= "Restored "+ nLastRestore+ " times ago";
        return s;
    }

    private double f(double steps, double i)
    {
        return (steps/100000.)+(i/(double)timeOver);
    }

    /**
     * Creates food at a random location. Only one piece of food can be spawned at a time.
     */
    private void newFood () {
        Random rX = new Random();
        Random rY = new Random();
        food = new Square(
                Square.Entity.Food,
                rX.nextInt(Configuration.BOARD_COLUMNS),
                rY.nextInt(Configuration.BOARD_ROWS));

        // If food is spawned inside the snake, try spawning it elsewhere.
        if (snake.contains(food)) {
            newFood();
        }
    }

    /**
     * Sets the direction of the Snake to go left.
     */
    void directionLeft () {
        if (lastMove != Direction.RIGHT || getSnakeSize() == 1) {
            movement = Direction.LEFT;
        }
    }

    /**
     * Sets the direction of the Snake to go right.
     */
    void directionRight () {
        if (lastMove != Direction.LEFT || getSnakeSize() == 1) {
            movement = Direction.RIGHT;
        }
    }

    /**
     * Sets the direction of the Snake to go up.
     */
    void directionUp () {
        if (lastMove != Direction.DOWN || getSnakeSize() == 1) {
            movement = Direction.UP;
        }
    }

    /**
     * Sets the direction of the Snake to go down.
     */
    void directionDown () {
        if (lastMove != Direction.UP || getSnakeSize() == 1) {
            movement = Direction.DOWN;
        }
    }

    /**
     * Moves the Snake one square, according to its direction.
     */
    private void moveSnake () {


        if (movement == Direction.LEFT) {
            moveSnakeLeft();
        } else if (movement == Direction.RIGHT) {
            moveSnakeRight();
        } else if (movement == Direction.UP) {
            moveSnakeUp();
        } else if (movement == Direction.DOWN) {
            moveSnakeDown();
        }
        lastMove = movement;

    }

    private void savePlayer()
    {
        if (fitness>0.003)
        {
            new Player(Configuration.goodPlayersDirectory,brain);
        }
    }

    private void moveSnakeLeft ()
    {
        if (!snake.moveLeft())
        {
            // Check to see if the Snake has run into itself.
            reset();
        }
        checkBounds();
        checkIfAteFood();
        movement = Direction.LEFT;
    }

    private void moveSnakeRight ()
    {
        if (!snake.moveRight()) // Check to see if the Snake has run into itself.
        {
            reset();
        }
        checkBounds();
        checkIfAteFood();
        movement = Direction.RIGHT;
    }

    private void moveSnakeUp () {
        if (!snake.moveUp()) { // Check to see if the Snake has run into itself.
            reset();
        }
        checkBounds();
        checkIfAteFood();
        movement = Direction.UP;
    }

    private void moveSnakeDown () {
        if (!snake.moveDown()) { // Check to see if the Snake has run into itself.
            reset();
        }
        checkBounds();
        checkIfAteFood();
        movement = Direction.DOWN;
    }

    private void checkBounds () {
        Square sq = snake.getHead();

        boolean tooFarLeft = sq.getX() < 0;
        boolean tooFarRight = sq.getX() >= Configuration.BOARD_COLUMNS;
        boolean tooFarUp = sq.getY() < 0;
        boolean tooFarDown = sq.getY() >= Configuration.BOARD_ROWS;

        if (tooFarLeft || tooFarRight || tooFarUp || tooFarDown) {
            reset();
        }
    }

    private void checkIfAteFood() {
        if (isSnakeOnFood()) {
            growSnake();
            newFood();
        }
    }

    private int getSnakeSize () {
        return snake.getSize();
    }

    void reset()
    {

        if (fitness==bestFitness)
        {
            if(goodPlayers<maxP)
            {
                population[goodPlayers] =new Player(brain,fitness);
                goodPlayers++;
            }
            else sort();
        }

        if (nLastGood>5)
        {
            nLastGood=0;
            resetWeights();
        }
        else
        {
            mutate();
            nLastGood ++;
        }

        if (fitness == bestFitness)
            savePlayer();
        this.snake = new Snake();

        this.score = 0;
        this.fitness = 0;
        timeOver = 500;
        newFood();
        update();
    }

    private void restore()
    {
        if (nLastRestore<5)
        {
            brain = population[new Random().nextInt(goodPlayers)].getModel();
            nLastRestore++;
        }
        else
        {
            nLastRestore = 0;
            resetWeights();
        }
    }

    static void mutate()
    {
        if (goodPlayers<maxP)
        {
            resetWeights();
        }
        else
        {
            Map<String, INDArray> parent1 = population[new Random().nextInt(3)].getModel().paramTable();
            Map<String, INDArray> parent2 = population[new Random().nextInt(3)].getModel().paramTable();
            Set<String> keys = parent1.keySet();
            for (String key : keys)
            {
                INDArray values;
                if (new Random().nextDouble()<5)
                    values = parent1.get(key);
                else
                    values = parent2.get(key);

                if (new Random().nextDouble()<0.1)
                    values = Nd4j.rand(values.shape());
                brain.setParam(key, values);
            }

        }
    }

    static void resetWeights()
    {
        Map<String, INDArray> paramTable = brain.paramTable();
        Set<String> keys = paramTable.keySet();
        for (String key : keys) {
            INDArray values = paramTable.get(key);
            brain.setParam(key, Nd4j.rand(values.shape()));//set some random values
        }
    }
    void rs()
    {
        resetWeights();
        this.snake = new Snake();
        this.score = 0;
        this.fitness = 0;
        timeOver = 500;
        newFood();
        update();
    }

    int getScore () {
        return score;
    }

    private boolean isSnakeOnFood () {
        return snake.getHead().equals(food);
    }

    private void growSnake () {
        snake.grow();
        score += 10;
    }

    void paint (Graphics graphics) {

        Graphics2D g = (Graphics2D) graphics;
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        //paintView(g);
        //paintCrossView(g);
        paintSnake(g);
        paintFood(g);
    }

    private void paintSnake (Graphics2D g) {
        int x, y;
        int corner = Configuration.SQUARE_SIZE / 3;

        for (Square sq : snake)
        {

            x = sq.getX() * Configuration.SQUARE_SIZE;
            y = sq.getY() * Configuration.SQUARE_SIZE;

            g.setColor(Configuration.snakeColor);
            g.fillRoundRect(x + 1, y + 1, Configuration.SQUARE_SIZE - 2,
                    Configuration.SQUARE_SIZE - 2, corner, corner);

        }

    }
    private void paintView(Graphics2D g)
    {

        int off = (Configuration.SQUARE_SIZE/2);
        int headX = (snake.getHead().getX()* Configuration.SQUARE_SIZE)+off, headY = (snake.getHead().getY()* Configuration.SQUARE_SIZE)+off;
        int ULX = 0,ULY = 0,URX = (Configuration.SQUARE_SIZE* Configuration.BOARD_COLUMNS),URY = 0 ;
        int DLX = 0, DLY= (Configuration.SQUARE_SIZE* Configuration.BOARD_ROWS) , DRX = URX  , DRY = DLY ;

        g.setColor(Color.RED);

        double head_UL = distance(g,new Point2D.Double(ULX,ULY));
        double head_UR = distance(g,new Point2D.Double(URX,URY));
        double head_DL = distance(g,new Point2D.Double(DLX,DLY));
        double head_DR = distance(g,new Point2D.Double(DRX,DRY));
        double head_FOOD = distance(g,new Point2D.Double((food.getX()* Configuration.SQUARE_SIZE)+off,(food.getY()* Configuration.SQUARE_SIZE)+off));
        visualfield[0] = head_UL;
        visualfield[1] = head_UR;
        visualfield[2] = head_DL;
        visualfield[3] = head_DR;
        visualfield[4] = head_FOOD;
    }

    private double distance(Graphics2D g ,Point2D toPoint)
    {
        int off = (Configuration.SQUARE_SIZE/2);
        int headX = (snake.getHead().getX()* Configuration.SQUARE_SIZE)+off, headY = (snake.getHead().getY()* Configuration.SQUARE_SIZE)+off;
        Point2D.Double headPoint = new Point2D.Double(headX,headY);
        Line2D.Double line = new Line2D.Double(headPoint,toPoint);
        g.draw(line);
        if (line.intersects(new Rectangle2D.Double(food.getX(),food.getY(), Configuration.SQUARE_SIZE, Configuration.SQUARE_SIZE)))
            return headPoint.distance(new Point2D.Double(food.getX(),food.getY()));
        for (Square sq: snake)
        {
            if (line.intersects(new Rectangle2D.Double(sq.getX(),sq.getY(), Configuration.SQUARE_SIZE, Configuration.SQUARE_SIZE)))
            {
                return headPoint.distance(new Point2D.Double(sq.getX(),sq.getY()));
            }
        }
        return headPoint.distance(toPoint);
    }

    private void paintCrossView(Graphics2D g)
    {
        int off = (Configuration.SQUARE_SIZE/2);
        int headX = (snake.getHead().getX()* Configuration.SQUARE_SIZE)+off, headY = (snake.getHead().getY()* Configuration.SQUARE_SIZE)+off;

        double head_UP = distance(g,new Point2D.Double(headX,0));
        double head_DOWN = distance(g,new Point2D.Double(headX,(Configuration.SQUARE_SIZE* Configuration.BOARD_ROWS)));
        double head_LEFT = distance(g,new Point2D.Double(0,headY));
        double head_RIGHT = distance(g,new Point2D.Double((Configuration.SQUARE_SIZE* Configuration.BOARD_COLUMNS),headY));
        visualfield[5] = head_UP;
        visualfield[6] = head_DOWN;
        visualfield[7] = head_RIGHT;
        visualfield[8] = head_LEFT;
    }


    private void paintFood (Graphics2D g)
    {
        int x = food.getX() * Configuration.SQUARE_SIZE;
        int y = food.getY() * Configuration.SQUARE_SIZE;
        int corner = Configuration.SQUARE_SIZE /3;

        g.setColor(Configuration.foodColor);
        g.fillRoundRect(x + 1, y + 1, Configuration.SQUARE_SIZE - 2,
                Configuration.SQUARE_SIZE - 2, corner, corner);
    }

    @Override
    public String toString () {

        StringBuilder sb = new StringBuilder();

        for (int y = 0; y < Configuration.BOARD_ROWS; y++) {
            for (int x = 0; x < Configuration.BOARD_COLUMNS; x++) {
                Square sq = new Square(x, y);

                if (snake.contains(sq)) {
                    sb.append("S");
                } else if (food.equals(sq)) {
                    sb.append("F");
                } else {
                    sb.append("-");
                }

                sb.append(" ");

            }
            sb.append("\n");
        }

        return new String(sb);
    }

    public void setField()
    {
        double [] field = new double[Configuration.BOARD_ROWS* Configuration.BOARD_COLUMNS];
        for (int y = 0; y < Configuration.BOARD_ROWS; y++)
        {
            int offset = y* Configuration.BOARD_ROWS;
            for (int x = 0; x < Configuration.BOARD_COLUMNS; x++)
            {
                int place = offset+x;
                Square sq = new Square(x, y);
                if (snake.contains(sq)) {
                    field[place]=.4;
                } else if (food.equals(sq)) {
                    field[place]=10.;
                } else {
                    field[place]= 0;
                }
            }
        }
        this.visualfield = field;
    }
}
