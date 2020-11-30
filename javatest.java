
import java.awt.Graphics;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class javatest extends JPanel {

    public void paint(Graphics g) {
        g.drawLine(10, 10, 300, 300);
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("javatesting ip");
        frame.getContentPane().add(new javatest());
        frame.setSize(1000, 1000);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setResizable(false);

    }

}

class Javatest{

    public static void main(String[] args){

        System.out.println("hello world");
    }

}

