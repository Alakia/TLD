#include <opencv2/opencv.hpp>
#include "tld_utils.h"
#include "TLD.h"
#include <iostream>
#include <sstream>
#include <stdio.h>
using namespace cv;
using namespace std;
//Global variables
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
string video;

void readBB(char* file){
	ifstream bb_file (file);
	string line;
	getline(bb_file,line);
	istringstream linestream(line);
	string x1,y1,x2,y2;
	getline (linestream,x1, ',');
	getline (linestream,y1, ',');
	getline (linestream,x2, ',');
	getline (linestream,y2, ',');
	int x = atoi(x1.c_str());// = (int)file["bb_x"];
	int y = atoi(y1.c_str());// = (int)file["bb_y"];
	int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
	int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
	box = Rect(x,y,w,h);
}
//bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param)
{
	switch( event )
	{
	case CV_EVENT_MOUSEMOVE:
		if (drawing_box)
		{
			box.width = x-box.x;
			box.height = y-box.y;
		}
		break;
	case CV_EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = Rect( x, y, 0, 0 );
		break;
	case CV_EVENT_LBUTTONUP:
		drawing_box = false;
		if( box.width < 0 )
		{
			box.x += box.width;
			box.width *= -1;
		}
		if( box.height < 0 )
		{
			box.y += box.height;
			box.height *= -1;
		}
		gotBB = true;
		break;
	}
}

void print_help(char** argv){
	printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
	printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}

void read_options(int argc, char** argv,VideoCapture& capture,FileStorage &fs){
	for (int i=0;i<argc;i++){
		if (strcmp(argv[i],"-b")==0){
			if (argc>i){
				readBB(argv[i+1]);
				gotBB = true;
			}
			else
				print_help(argv);
		}
		if (strcmp(argv[i],"-s")==0){
			if (argc>i){
				video = string(argv[i+1]);
				capture.open(video);
				fromfile = true;
			}
			else
				print_help(argv);

		}
		if (strcmp(argv[i],"-p")==0){
			if (argc>i){
				fs.open(argv[i+1], FileStorage::READ);
			}
			else
				print_help(argv);
		}
		if (strcmp(argv[i],"-no_tl")==0){
			tl = false;
		}
		if (strcmp(argv[i],"-r")==0){
			rep = true;
		}
	}
}

int main(int argc, char * argv[])
{
	VideoCapture capture;
	capture.open(0);
	FileStorage fs;
	VideoWriter vdo_writer;
	Size vdo_size;
	double fps; // Ö¡ÂÊ
	//const char out_file[200] = "/guest/stab/car.mpg"; //

	// ¶ÁÈëÃüÁîÐÐ²ÎÊý
	//read_options(argc,argvcapture.open(),capture,fs);

	//Init camera
	//if (!capture.isOpened())
	//{
	//	cout << "capture device failed to open!" << endl;
	//	return 1;
	//}

	// ×¢²áÊó±êµÄ»Øµ÷º¯Êý
	cvNamedWindow("TLD",CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback( "TLD", mouseHandler, NULL );

	//TLD framework
	TLD tld;

	// ¶Á parameters.yml ÎÄŒþ
	tld.read(fs.getFirstTopLevelNode());

	tld.min_win = 10;
	tld.patch_size = 15;
	tld.num_closest_init = 10;
	tld.num_warps_init = 20;
	tld.noise_init = 5;
	tld.angle_init = 20;
	tld.shift_init = 0.02;
	tld.scale_init = 0.02;
	tld.num_closest_update = 10;
	tld.num_warps_update = 10;
	tld.noise_update = 5;
	tld.angle_update = 10;
	tld.shift_update = 0.02;
	tld.scale_update = 0.02;
	tld.bad_overlap = 0.2;
	tld.bad_patches = 100;


	tld.classifier.valid = 0.5;
	tld.classifier.ncc_thesame = 0.95;
	tld.classifier.nstructs = 10;
	tld.classifier.structSize = 13;
	tld.classifier.thr_fern = 0.6;
	tld.classifier.thr_nn = 0.65;
	tld.classifier.thr_nn_valid = 0.7;



	Mat frame;
	Mat Expand;
	Mat last_gray;
	Mat first;
	if (fromfile)
	{
		capture >> frame; // œ«µÚÒ»Ö¡
		cvtColor(frame, last_gray, CV_RGB2GRAY);
		frame.copyTo(first);
	}
	else
	{
		capture.set(CV_CAP_PROP_FRAME_WIDTH,320);
		capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
	}

	fps = capture.get(CV_CAP_PROP_FPS);
	vdo_size = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH),
					(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	//vdo_writer.open(out_file, -1, fps, vdo_size, true); //CV_FOURCC_DEFAULT PIM1 MJPG  CV_FOURCC('P','I','M','1')

	///Initialization
GETBOUNDINGBOX:
	while(!gotBB) // Èç¹û»¹Ã»ÓÐ»ñµÃ³õÊŒÎïÌåµÄbounding box£¬ÔòµÈŽý
	{
		if (!fromfile)
		{
			capture >> frame;
		}
		else // ŽÓÎÄŒþ
			first.copyTo(frame);

		cvtColor(frame, last_gray, CV_RGB2GRAY);
		drawBox(frame,box);
		imshow("TLD", frame);
		if (cvWaitKey(33) == 'q')
			return 0;
	}


	//test
		//box.x = 80;
		//box.y = 70;
		//box.width = 30;
		//box.height = 30;

	// ÒÑŸ­»ñÈ¡µÚÒ»Ö¡ÍŒÏñÖÐµÄÄ¿±ê
	if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"])
	{// ÈË¹€Ñ¡µÄÄ¿±êŽóÐ¡²»ÄÜÐ¡ÓÚÔ€ÏÈÉè¶šµÄÏÂÏÞ
		cout << "Bounding box too small, try again." << endl;
		gotBB = false;
		goto GETBOUNDINGBOX;
	}
	// È¥³ýÊó±ê»Øµ÷º¯Êý¹ŠÄÜ
	cvSetMouseCallback( "TLD", NULL, NULL );



	printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);

	//œ«Ñ¡¶šµÄÄ¿±êŸØÐÎÊä³öµœÍâ²¿ÎÄŒþÖÐ
	//FILE  *bb_file = fopen("C:\\Users\\Administrator\\Desktop\\TLD_HY-2012-11-27\\TLD_HY\\bounding_boxes.txt","w");
	FILE  *bb_file = fopen("/home/guest/TLD_HY/TLD_HY/bounding_boxes.txt","w");

	//TLD Ëã·š³õÊŒ»¯
	tld.init(last_gray, box, bb_file);

	///Run-time
	Mat current_gray;
	BoundingBox pbox;
	vector<Point2f> pts1;
	vector<Point2f> pts2;
	bool status=true;
	int frames = 1;
	int detections = 1;

	int c;
	double t1, t2;

REPEAT:
	// ¶ÁÈëÐÂµÄÒ»Ö¡ÍŒÏñ
	while(capture.read(frame))
	{
		//œ«²ÊÉ«frame×ªÎª»Ò¶Ècurrent_gray
		cvtColor(frame, current_gray, CV_RGB2GRAY);
		/*
		ÖðÖ¡¶ÁÈëÍŒÆ¬ÐòÁÐ£¬œøÐÐËã·šŽŠÀí¡£processFrame¹²°üº¬ËÄžöÄ£¿é£šÒÀŽÎŽŠÀí£©£º
		žú×ÙÄ£¿é¡¢Œì²âÄ£¿é¡¢×ÛºÏÄ£¿éºÍÑ§Ï°Ä£¿é
		*/
		t1 = cv::getTickCount();
		tld.processFrame(last_gray,current_gray,pts1,pts2,pbox,status,tl,bb_file);
		t2 = (cv::getTickCount() - t1) / cv::getTickFrequency();
		printf("timecount is %f \n", t2);

		//Draw Points
		if (status)
		{
			//drawPoints(frame,pts1);
			//drawPoints(frame,pts2,Scalar(0,255,0));
			drawBox(frame,pbox, Scalar(0,0,2550), 3);
			detections++;
		}
		//Display
		//Expand = frame.reshape(640, 480);
		Size sz(640, 480);
		cv::pyrUp(frame, Expand, sz);

		imshow("TLD", Expand);

		// ±£Žæ³ÉÊÓÆµÎÄŒþ
		vdo_writer << frame;

		//swap points and images
		swap(last_gray,current_gray);
		pts1.clear();
		pts2.clear();
		frames++;
		printf("Detection rate: %d/%d\n",detections,frames);

		if (cvWaitKey(33) == 'q')
			break;
	}
	if (rep)
	{
		rep = false;
		tl = false;
		fclose(bb_file);
		bb_file = fopen("final_detector.txt","w");

		//capture.set(CV_CAP_PROP_POS_AVI_RATIO,0);
		capture.release();
		capture.open(video);
		goto REPEAT;
	}
	fclose(bb_file);

	return 0;
}
