package src.game;

import java.awt.*;

/**
 * Contains the details and parameters of the game, as well as some presets.
 */
public class Configuration {

    // GameBoard.
    static final int BOARD_COLUMNS   = 30;
    static final int BOARD_ROWS      = 20;
    static final int SQUARE_SIZE     = 25;

    // Snake.
    static final int START_X         = BOARD_COLUMNS / 2;
    static final int START_Y         = BOARD_ROWS / 2;

    // Colors.
    static Color backgroundColor    = new Color(53, 53, 53);
    static Color snakeColor         = new Color(0, 255, 255);
    static Color foodColor          = new Color(211, 87, 45);
    
    // 7/27/2017
    // Add constant color changing background
    static Color green = new Color(1, 219, 129);
    static Color blue = new Color(31, 107, 255);
    static Color violet = new Color(150, 62, 238);
    static Color red = new Color(230, 61, 61);
    static Color orange = new Color(255, 147, 41);
    static Color yellow = new Color(250, 247, 95);
    
    static Color[] colors = {green, blue, violet, red, orange, yellow};
    
    static Color currentRGB = colors[0];
    
    static Color nextColorRGB;
    
    static boolean newColor = true;
    static int index = 0;
    static Theme theme = Theme.Dark;

    static  String goodPlayersDirectory = "models/0/model"+ System.goodPlayers+".zip";
    
    static void changeColor()
    {
        if (newColor) {
            if (index < colors.length - 1) index++;
            else index = 0;
            nextColorRGB = colors[index];
            newColor = false;
        }
        
        boolean change = false;

        if (currentRGB.getRed() > nextColorRGB.getRed()) {
            currentRGB = new Color(currentRGB.getRed() - 1, currentRGB.getGreen(), currentRGB.getBlue()); 
            change = true;
        }
        else if (currentRGB.getRed() < nextColorRGB.getRed()) {
            currentRGB = new Color(currentRGB.getRed() + 1, currentRGB.getGreen(), currentRGB.getBlue()); 
            change = true;
        }
        if (currentRGB.getGreen() > nextColorRGB.getGreen()) {
            currentRGB = new Color(currentRGB.getRed(), currentRGB.getGreen() - 1, currentRGB.getBlue()); 
            change = true;
        }
        else if (currentRGB.getGreen() < nextColorRGB.getGreen()) {
            currentRGB = new Color(currentRGB.getRed(), currentRGB.getGreen() + 1, currentRGB.getBlue()); 
            change = true;
        }
        if (currentRGB.getBlue() > nextColorRGB.getBlue()) {
            currentRGB = new Color(currentRGB.getRed(), currentRGB.getGreen(), currentRGB.getBlue() - 1); 
            change = true;
        }
        else if (currentRGB.getBlue() < nextColorRGB.getBlue()) {
            currentRGB = new Color(currentRGB.getRed(), currentRGB.getGreen(), currentRGB.getBlue() + 1); 
            change = true;
        }

        if (!change) newColor = true;
        
        Configuration.backgroundColor = currentRGB;
    }
    
    static Theme getTheme() {
        return theme;
    }
    
    static void Rainbow () {
        Configuration.backgroundColor = colors[0];
        Configuration.theme = Theme.Rainbow;
    }

    static void Dark () {
        Configuration.backgroundColor = new Color(53, 53, 53);
        Configuration.snakeColor = new Color(0, 254, 254);
        Configuration.foodColor = new Color(211, 87, 45);
        Configuration.theme = Theme.Dark;
    }

    static void Sky () {
        Configuration.backgroundColor = new Color(25, 181, 254);
        Configuration.snakeColor = new Color(255, 255, 255);
        Configuration.foodColor = new Color(0, 119, 192);
        Configuration.theme = Theme.Sky;
    }

    static void Mud () {
        Configuration.backgroundColor = new Color(94, 44, 11);
        Configuration.snakeColor = new Color(246, 196, 163);
        Configuration.foodColor = new Color(211, 87, 45);
        Configuration.theme = Theme.Mud;
    }

    static void Sand () {
        Configuration.backgroundColor = new Color(253, 227, 167);
        Configuration.snakeColor = new Color(142, 68, 173);
        Configuration.foodColor = new Color(243, 156, 18);
        Configuration.theme = Theme.Sand;
    }
    
    enum Theme {
        Rainbow,
        Dark,
        Sky,
        Mud,
        Sand
    }

}