package application;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;


import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

public class FDxmlController {

	@FXML
	private CheckBox haarClassifier;

	@FXML
	private CheckBox lbpClassifier;

	@FXML
	private Button cameraButton;

	@FXML
	private ImageView originalFrame;

	private ScheduledExecutorService timer;
    private VideoCapture capture;
    private boolean cameraActive;
    private CascadeClassifier faceCascade;
    private int absoluteFaceSize;

    protected void init()
    {
    	this.capture = new VideoCapture();
    	this.faceCascade = new CascadeClassifier();
    	this.absoluteFaceSize=0;
    }

    @FXML
    protected void startCamera()
    {
    	originalFrame.setFitWidth(600);
    	originalFrame.setPreserveRatio(true);

    	if(!this.cameraActive)
    	{
    		this.haarClassifier.setDisable(true);
    		this.lbpClassifier.setDisable(true);
    		this.capture.open(0);

    		if(this.capture.isOpened())
    		{
    			this.cameraActive = true;
    			Runnable frameGrabber = new Runnable() {
    				@Override
    				public void run()
    				{
    					Image imageToShow = grabFrame();
    					originalFrame.setImage(imageToShow);
    				}
    			};

    			this.timer = Executors.newSingleThreadScheduledExecutor();
    			this.timer.scheduleAtFixedRate(frameGrabber,0,33,TimeUnit.MILLISECONDS);

    			this.cameraButton.setText("Stop");
    		}

    		else
    		{
    			System.err.println("Cannot open camera");
    		}

    	}

    	else
    	{
    		this.cameraActive = false;
    		this.cameraButton.setText("Start");
    		this.haarClassifier.setDisable(false);
    		this.lbpClassifier.setDisable(false);

    		try
    		{
    			this.timer.shutdown();
    			this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
    		}
    		catch(InterruptedException e)
    		{
    			System.err.println("Exception in stopping frame capture");
    		}

    		this.capture.release();
    		this.originalFrame.setImage(null);
    	}
    }

    private Image grabFrame()
    {
    	Image imageToShow = null;
    	Mat frame = new Mat();

    	if(this.capture.isOpened())
    	{
    		try
    		{
    			this.capture.read(frame);
    			if(!frame.empty())
    			{
    				this.detectAndDisplay(frame);
    				imageToShow = mat2Image(frame);
    			}
    		}
    		catch(Exception e)
    		{
    			System.err.println("Error in grabFrame() "+e);
    		}
    	}

    	return imageToShow;

    }

    private void detectAndDisplay(Mat frame)
    {
    	MatOfRect faces = new MatOfRect();
    	Mat grayFrame = new Mat();

    	Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
    	Imgproc.equalizeHist(grayFrame,grayFrame);

    	if(this.absoluteFaceSize==0)
    	{
    		int height = grayFrame.rows();
    		if(Math.round(height * 0.2f)>0)
    		{
    			this.absoluteFaceSize = Math.round(height * 0.2f);

    		}
    	}

    	this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
    		new Size(this.absoluteFaceSize,this.absoluteFaceSize),new Size());

    	Rect[] facesArray = faces.toArray();
    	for(int i = 0;i<facesArray.length;i++)
    	{
    		Imgproc.rectangle(frame, facesArray[i].tl(),facesArray[i].br(), new Scalar(0,255,0),3);
    	}
    }

    @FXML
    protected void haarSelected(Event event)
    {
    	if(this.lbpClassifier.isSelected())
    	{
    		this.lbpClassifier.setSelected(false);
    	}

    	this.checkboxSelection("C:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml");

    }

    @FXML
    protected void lbpSelected(Event event)
    {
    	if(this.haarClassifier.isSelected())
    	{
    		this.haarClassifier.setSelected(false);
    	}

    	this.checkboxSelection("C:/opencv/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");

    }

    private void checkboxSelection(String classifierPath)
    {
    	this.faceCascade.load(classifierPath);
    	this.cameraButton.setDisable(false);
    }

    private Image mat2Image(Mat frame)
    {
    	MatOfByte buffer = new MatOfByte();
    	Imgcodecs.imencode(".png",frame,buffer);
    	return new Image(new ByteArrayInputStream(buffer.toArray()));
    }

}