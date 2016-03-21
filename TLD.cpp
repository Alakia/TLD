/*
* TLD.cpp
*
*  Created on: Jun 9, 2011
*      Author: alantrrs
*/

#include "TLD.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;//ÃüÃû¿ÕŒä


TLD::TLD()
{
}
TLD::TLD(const FileNode& file){
	read(file);
}
TLD::~TLD()
{
	std::vector<BoundingBox*>::iterator pos;
	for(pos=grid.begin(); pos!=grid.end(); pos++)
	{
		delete *pos;
	}
	grid.clear();
}

void TLD::read(const FileNode& file)
{
	///Bounding Box Parameters
	min_win = (int)file["min_win"];
	///Genarator Parameters
	//initial parameters for positive examples
	patch_size = (int)file["patch_size"];
	num_closest_init = (int)file["num_closest_init"];
	num_warps_init = (int)file["num_warps_init"];
	noise_init = (int)file["noise_init"];
	angle_init = (float)file["angle_init"];
	shift_init = (float)file["shift_init"];
	scale_init = (float)file["scale_init"];
	//update parameters for positive examples
	num_closest_update = (int)file["num_closest_update"];
	num_warps_update = (int)file["num_warps_update"];
	noise_update = (int)file["noise_update"];
	angle_update = (float)file["angle_update"];
	shift_update = (float)file["shift_update"];
	scale_update = (float)file["scale_update"];
	//parameters for negative examples
	bad_overlap = (float)file["overlap"];
	bad_patches = (int)file["num_patches"];
	classifier.read(file);
}

void TLD::init(const Mat& frame1,const Rect& box, FILE* bb_file)
{
	//bb_file = fopen("bounding_boxes.txt","w");
	//Get Bounding Boxes

	/* 1. ŽŽœš¶àžö³ß¶ÈµÄbounding box
	Œì²âÆ÷²ÉÓÃÉšÃèŽ°¿ÚµÄ²ßÂÔ£ºÉšÃèŽ°¿Ú²œ³€Îª¿ížßµÄ 10%£¬³ß¶ÈËõ·ÅÏµÊýÎª1.2£»ŽËº¯Êý¹¹œšÈ«²¿µÄÉšÃèŽ°¿Úgrid£¬
	²¢ŒÆËãÃ¿Ò»žöÉšÃèŽ°¿ÚÓëÊäÈëµÄÄ¿±êboxµÄÖØµþ¶È£»ÖØµþ¶È¶šÒåÎªÁœžöboxµÄœ»Œ¯ÓëËüÃÇµÄ²¢Œ¯µÄ±È
	*/
	buildGrid(frame1,box);


	///Preparation
	//2. Îªž÷ÖÖ±äÁ¿»òÕßÈÝÆ÷·ÖÅäÄÚŽæ¿ÕŒä
	iisum.create(frame1.rows+1, frame1.cols+1, CV_32F);
	iisqsum.create(frame1.rows+1,frame1.cols+1,CV_64F);
	dconf.reserve(100);
	dbb.reserve(100);
	bbox_step =7;
	//tmp.conf.reserve(grid.size());
	tmp.conf = vector<float>(grid.size());
	tmp.patt = vector<vector<int> >(grid.size(),vector<int>(10,0));
	//tmp.patt.reserve(grid.size());
	dt.bb.reserve(grid.size());
	good_boxes.reserve(grid.size());
	bad_boxes.reserve(grid.size());
	pEx.create(patch_size,patch_size,CV_64F);
	// ³õÊŒ»¯ Generator
	generator = PatchGenerator (0,0,noise_init,true,1-scale_init,1+scale_init,-angle_init*CV_PI/180,angle_init*CV_PI/180,-angle_init*CV_PI/180,angle_init*CV_PI/180);

	getOverlappingBoxes(box,10);//numclosestinit to 10

	Rect temp;
		for(int pi = 0; pi < (int)good_boxes.size(); pi++)
		{
			temp.x = grid[good_boxes[pi]]->x;
			temp.y = grid[good_boxes[pi]]->y;
			temp.width = grid[good_boxes[pi]]->width;
			temp.height = grid[good_boxes[pi]]->height;
			printf("%d %d, %d, %d, %d \n", pi, temp.x, temp.y, temp.width, temp.height);
			//rectangle(frame1, temp, CV_RGB(255,255,255), 1, CV_AA, 0);
			//imshow("frame1", frame1);
			//waitKey(0);
		}

		printf("Created %d bounding boxes\n",(int)grid.size());

	printf("Found %d good boxes, %d bad boxes\n",(int)good_boxes.size(),(int)bad_boxes.size());
	printf("Best Box: %d %d %d %d\n",best_box.x,best_box.y,best_box.width,best_box.height);
	printf("Bounding box hull: %d %d %d %d\n",bbhull.x,bbhull.y,bbhull.width,bbhull.height);
	//Correct Bounding Box
	lastbox=best_box;
	lastconf=1;
	lastvalid=true;
	//Print
	fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);

	/*
	4 ×Œ±ž·ÖÀàÆ÷£¬scalesÈÝÆ÷ÀïÊÇËùÓÐÉšÃèŽ°¿ÚµÄ³ß¶È£¬ÓÉÉÏÃæµÄbuildGrid()º¯Êý³õÊŒ»¯£»
	TLDµÄ·ÖÀàÆ÷ÓÐÈý²¿·Ö£º·œ²î·ÖÀàÆ÷Ä£¿é¡¢Œ¯ºÏ·ÖÀàÆ÷Ä£¿éºÍ×îœüÁÚ·ÖÀàÆ÷Ä£¿é£»ÕâÈýžö·ÖÀàÆ÷ÊÇŒ¶ÁªµÄ£¬Ã¿Ò»žöÉšÃèŽ°¿ÚÒÀŽÎÈ«²¿Íš¹ýÉÏÃæÈýžö·ÖÀàÆ÷£¬
	²Å±»ÈÏÎªº¬ÓÐÇ°Ÿ°Ä¿±ê¡£ÕâÀïprepareÕâžöº¯ÊýÖ÷ÒªÊÇ³õÊŒ»¯Œ¯ºÏ·ÖÀàÆ÷Ä£¿é£»Œ¯ºÏ·ÖÀàÆ÷£šËæ»úÉ­ÁÖ£©»ùÓÚnžö»ù±Ÿ·ÖÀàÆ÷£š¹²10¿ÃÊ÷£©£¬
	Ã¿žö·ÖÀàÆ÷£šÊ÷£©¶ŒÊÇ»ùÓÚÒ»žöpixel comparisons£š¹²13žöÏñËØ±ÈœÏŒ¯£©µÄ£¬Ò²ŸÍÊÇËµÃ¿¿ÃÊ÷ÓÐ13žöÅÐ¶ÏœÚµã£š×é³ÉÒ»žöpixel comparisons£©£¬
	ÊäÈëµÄÍŒÏñÆ¬ÓëÃ¿Ò»žöÅÐ¶ÏœÚµã£šÏàÓŠÏñËØµã£©œøÐÐ±ÈœÏ£¬²úÉú0»òÕß1£¬È»ºóœ«Õâ13žö0»òÕß1Á¬³ÉÒ»žö13Î»µÄ¶þœøÖÆÂëx£šÓÐ2^13ÖÖ¿ÉÄÜ£©£¬
	Ã¿Ò»žöx¶ÔÓŠÒ»žöºóÑéžÅÂÊP(y|x)= #p/(#p+#n) £šÒ²ÓÐ2^13ÖÖ¿ÉÄÜ£©£¬#pºÍ#n·Ö±ðÊÇÕýºÍžºÍŒÏñÆ¬µÄÊýÄ¿¡£ÄÇÃŽÕûÒ»žöŒ¯ºÏ·ÖÀàÆ÷£š¹²10žö»ù±Ÿ·ÖÀàÆ÷£©ŸÍÓÐ10žöºóÑéžÅÂÊÁË£¬
	œ«10žöºóÑéžÅÂÊœøÐÐÆœŸù£¬Èç¹ûŽóÓÚãÐÖµ£šÒ»¿ªÊŒÉèŸ­ÑéÖµ0.65£¬ºóÃæÔÙÑµÁ·ÓÅ»¯£©µÄ»°£¬ŸÍÈÏÎªžÃÍŒÏñÆ¬º¬ÓÐÇ°Ÿ°Ä¿±ê£»
	ºóÑéžÅÂÊP(y|x)= #p/(#p+#n)µÄ²úÉú·œ·š£º³õÊŒ»¯Ê±£¬Ã¿žöºóÑéžÅÂÊ¶ŒµÃ³õÊŒ»¯Îª0£»ÔËÐÐÊ±ºòÒÔÏÂÃæ·œÊœžüÐÂ£ºœ«ÒÑÖªÀà±ð±êÇ©µÄÑù±Ÿ£šÑµÁ·Ñù±Ÿ£©Íš¹ýnžö·ÖÀàÆ÷œøÐÐ·ÖÀà£¬
	Èç¹û·ÖÀàœá¹ûŽíÎó£¬ÄÇÃŽÏàÓŠµÄ#pºÍ#nŸÍ»ážüÐÂ£¬ÕâÑùP(y|x)Ò²ÏàÓŠžüÐÂÁË¡£
	pixel comparisonsµÄ²úÉú·œ·š£ºÏÈÓÃÒ»žö¹éÒ»»¯µÄpatchÈ¥ÀëÉ¢»¯ÏñËØ¿ÕŒä£¬²úÉúËùÓÐ¿ÉÄÜµÄŽ¹Ö±ºÍË®ÆœµÄpixel comparisons£¬
	È»ºóÎÒÃÇ°ÑÕâÐ©pixel comparisonsËæ»ú·ÖÅäžønžö·ÖÀàÆ÷£¬Ã¿žö·ÖÀàÆ÷µÃµœÍêÈ«²»Í¬µÄpixel comparisons£šÌØÕ÷Œ¯ºÏ£©£¬
	ÕâÑù£¬ËùÓÐ·ÖÀàÆ÷µÄÌØÕ÷×éÍ³Ò»ÆðÀŽŸÍ¿ÉÒÔž²žÇÕûžöpatchÁË¡£
	ÌØÕ÷ÊÇÏà¶ÔÓÚÒ»ÖÖ³ß¶ÈµÄŸØÐÎ¿ò¶øÑÔµÄ£¬TLDÖÐµÚsÖÖ³ß¶ÈµÄµÚižöÌØÕ÷features[s][i] = Feature(x1, y1, x2, y2);
	ÊÇÁœžöËæ»ú·ÖÅäµÄÏñËØµã×ø±ê£šŸÍÊÇÓÉÕâÁœžöÏñËØµã±ÈœÏµÃµœ0»òÕß1µÄ£©¡£Ã¿Ò»ÖÖ³ß¶ÈµÄÉšÃèŽ°¿Ú¶Œº¬ÓÐtotalFeatures = nstructs * structSizežöÌØÕ÷£»
	nstructsÎªÊ÷ÄŸ£šÓÉÒ»žöÌØÕ÷×é¹¹œš£¬Ã¿×éÌØÕ÷Žú±íÍŒÏñ¿éµÄ²»Í¬ÊÓÍŒ±íÊŸ£©µÄžöÊý£»structSizeÎªÃ¿¿ÃÊ÷µÄÌØÕ÷žöÊý£¬Ò²ŒŽÃ¿¿ÃÊ÷µÄÅÐ¶ÏœÚµãžöÊý£»
	È»ºó³õÊŒ»¯ºóÑéžÅÂÊÎª0£»
	*/
	classifier.prepare(scales);

	/*
	5 Íš¹ý¶ÔµÚÒ»Ö¡ÍŒÏñµÄÄ¿±ê¿òbox£šÓÃ»§Öž¶šµÄÒªžú×ÙµÄÄ¿±ê£©œøÐÐ·ÂÉä±ä»»ÀŽºÏ³ÉÑµÁ·³õÊŒ·ÖÀàÆ÷µÄÕýÑù±ŸŒ¯¡£
	ŸßÌå·œ·šÈçÏÂ£ºÏÈÔÚŸàÀë³õÊŒµÄÄ¿±ê¿ò×îœüµÄÉšÃèŽ°¿ÚÄÚÑ¡Ôñ10žöbounding box£¬È»ºóÔÚÃ¿žöbounding boxµÄÄÚ²¿£¬œøÐÐ1%·¶Î§µÄÆ«ÒÆ£¬1%·¶Î§µÄ³ß¶È±ä»¯£¬10%·¶Î§µÄÆœÃæÄÚÐý×ª£¬
	²¢ÇÒÔÚÃ¿žöÏñËØÉÏÔöŒÓ·œ²îÎª5µÄžßË¹ÔëÉù£šÈ·ÇÐµÄŽóÐ¡ÊÇÔÚÖž¶šµÄ·¶Î§ÄÚËæ»úÑ¡ÔñµÄ£©£¬ÄÇÃŽÃ¿žöbox¶ŒœøÐÐ20ŽÎÕâÖÖŒžºÎ±ä»»£¬
	ÄÇÃŽ10žöboxœ«²úÉú200žö·ÂÉä±ä»»µÄbounding box£¬×÷ÎªÕýÑù±Ÿ¡£
	*/
	generatePositiveData(frame1,num_warps_init);

	/*
	6 Í³ŒÆbest_boxµÄŸùÖµºÍ±ê×Œ²î£¬var = pow(stdev.val[0],2) * 0.5;×÷Îª·œ²î·ÖÀàÆ÷µÄãÐÖµ
	*/
	Scalar stdev, mean;
	meanStdDev(frame1(best_box),mean,stdev);
	integral(frame1,iisum,iisqsum);
	var = pow(stdev.val[0],2)*0.5; //getVar(best_box,iisum,iisqsum);
	cout << "variance: " << var << endl;
	//check variance
	double vr =  getVar(best_box,iisum,iisqsum)*0.5;
	cout << "check variance: " << vr << endl;

	/*
	7 ÓÉÓÚTLDœöžú×ÙÒ»žöÄ¿±ê£¬ËùÒÔÎÒÃÇÈ·¶šÁËÄ¿±ê¿òÁË£¬¹Ê³ýÄ¿±ê¿òÍâµÄÆäËûÍŒÏñ¶ŒÊÇžºÑù±Ÿ£¬ÎÞÐè·ÂÉä±ä»»
	*/
	// Generate negative data
	generateNegativeData(frame1);

	/*
	8 œ«nExµÄÒ»°ë×÷ÎªÑµÁ·Œ¯nEx£¬ÁíÒ»°ë×÷Îª²âÊÔŒ¯nExT£»Í¬Ñù£¬nXÒ²²ð·ÖÎªÑµÁ·Œ¯nXºÍ²âÊÔŒ¯nXT
	*/
	int half = (int)nX.size()*0.5f;
	nXT.assign(nX.begin()+half,nX.end());
	nX.resize(half);
	///Split Negative NN Examples into Training and Testing sets
	half = (int)nEx.size()*0.5f;
	nExT.assign(nEx.begin()+half,nEx.end());
	nEx.resize(half);

	/*
	œ«žºÑù±ŸnXºÍÕýÑù±ŸpXºÏ²¢µœferns_data[]ÖÐ£¬ÓÃÓÚŒ¯ºÏ·ÖÀàÆ÷µÄÑµÁ·
	*/
	//Merge Negative Data with Positive Data and shuffle it
	vector<pair<vector<int>,int> > ferns_data(nX.size()+pX.size());
	vector<int> idx = index_shuffle(0, ferns_data.size());
	int a=0;
	for (int i=0;i<pX.size();i++)
	{
		ferns_data[idx[a]] = pX[i];
		a++;
	}
	for (int i=0;i<nX.size();i++)
	{
		ferns_data[idx[a]] = nX[i];
		a++;
	}
	/*
	œ«ÉÏÃæµÃµœµÄÒ»žöÕýÑù±ŸpExºÍnExºÏ²¢µœnn_data[]ÖÐ£¬ÓÃÓÚ×îœüÁÚ·ÖÀàÆ÷µÄÑµÁ·
	*/
	vector<cv::Mat> nn_data(nEx.size()+1);
	nn_data[0] = pEx;
	for (int i=0;i<nEx.size();i++)
	{
		nn_data[i+1]= nEx[i];
	}

	/*
	9 ÓÃÉÏÃæµÄÑù±ŸÑµÁ·Œ¯ÑµÁ· Œ¯ºÏ·ÖÀàÆ÷£šÉ­ÁÖ£© ºÍ ×îœüÁÚ·ÖÀàÆ÷
	*/
	///Training
	/*
	¶ÔÃ¿Ò»žöÑù±Ÿferns_data[i] £¬Èç¹ûÑù±ŸÊÇÕýÑù±Ÿ±êÇ©£¬ÏÈÓÃmeasure_forestº¯Êý·µ»ØžÃÑù±ŸËùÓÐÊ÷µÄËùÓÐÌØÕ÷Öµ¶ÔÓŠµÄºóÑéžÅÂÊÀÛŒÓÖµ£¬
	žÃÀÛŒÓÖµÈç¹ûÐ¡ÓÚÕýÑù±ŸãÐÖµ£š0.6* nstructs£¬ÕâŸÍ±íÊŸÆœŸùÖµÐèÒªŽóÓÚ0.6£š0.6* nstructs / nstructs£©,
	0.6ÊÇ³ÌÐò³õÊŒ»¯Ê±¶šµÄŒ¯ºÏ·ÖÀàÆ÷µÄãÐÖµ£¬ÎªŸ­ÑéÖµ£¬ºóÃæ»áÓÃ²âÊÔŒ¯ÀŽÆÀ¹ÀÐÞžÄ£¬ÕÒµœ×îÓÅ£©£¬Ò²ŸÍÊÇÊäÈëµÄÊÇÕýÑù±Ÿ£¬ÈŽ±»·ÖÀà³ÉžºÑù±ŸÁË£¬³öÏÖÁË·ÖÀàŽíÎó£¬
	ËùÒÔŸÍ°ÑžÃÑù±ŸÌíŒÓµœÕýÑù±Ÿ¿â£¬Í¬Ê±ÓÃupdateº¯ÊýžüÐÂºóÑéžÅÂÊ¡£¶ÔÓÚžºÑù±Ÿ£¬Í¬Ñù£¬Èç¹û³öÏÖžºÑù±Ÿ·ÖÀàŽíÎó£¬ŸÍÌíŒÓµœžºÑù±Ÿ¿â¡£
	*/
	classifier.trainF(ferns_data,2); //bootstrap = 2

	/*
	¶ÔÃ¿Ò»žöÑù±Ÿnn_data£¬Èç¹û±êÇ©ÊÇÕýÑù±Ÿ£¬Íš¹ýNNConf(nn_examples[i], isin, conf, dummy);ŒÆËãÊäÈëÍŒÏñÆ¬ÓëÔÚÏßÄ£ÐÍÖ®ŒäµÄÏà¹ØÏàËÆ¶Èconf£¬
	Èç¹ûÏà¹ØÏàËÆ¶ÈÐ¡ÓÚ0.65 £¬ÔòÈÏÎªÆä²»º¬ÓÐÇ°Ÿ°Ä¿±ê£¬Ò²ŸÍÊÇ·ÖÀàŽíÎóÁË£»ÕâÊ±ºòŸÍ°ÑËüŒÓµœÕýÑù±Ÿ¿â¡£
	È»ºóŸÍÍš¹ýpEx.push_back(nn_examples[i]);œ«žÃÑù±ŸÌíŒÓµœpExÕýÑù±Ÿ¿âÖÐ£»Í¬Ñù£¬Èç¹û³öÏÖžºÑù±Ÿ·ÖÀàŽíÎó£¬ŸÍÌíŒÓµœžºÑù±Ÿ¿â
	*/
	classifier.trainNN(nn_data);
	///Threshold Evaluation on testing sets
	/*
	ÓÃ²âÊÔŒ¯ÔÚÉÏÃæµÃµœµÄ Œ¯ºÏ·ÖÀàÆ÷£šÉ­ÁÖ£© ºÍ ×îœüÁÚ·ÖÀàÆ÷ÖÐ·ÖÀà£¬ÆÀŒÛ²¢ÐÞžÄµÃµœ×îºÃµÄ·ÖÀàÆ÷ãÐÖµ
	*/
	classifier.evaluateTh(nXT,nExT);
}

/* Generate Positive data
* Inputs:
* - good_boxes (bbP)
* - best_box (bbP0)
* - frame (im0)
* Outputs:
* - Positive fern features (pX)
* - Positive NN examples (pEx)
*/
void TLD::generatePositiveData(const Mat& frame, int num_warps)
{
	Scalar mean;
	Scalar stdev;
	/*
	1, ŽËº¯Êýœ«frameÍŒÏñbest_boxÇøÓòµÄÍŒÏñÆ¬¹éÒ»»¯ÎªŸùÖµÎª0µÄ15*15ŽóÐ¡µÄpatch£¬
	ŽæÓÚpEx£šÓÃÓÚ×îœüÁÚ·ÖÀàÆ÷µÄÕýÑù±Ÿ£©ÕýÑù±ŸÖÐ£š×îœüÁÚµÄboxµÄPattern£©£¬žÃÕýÑù±ŸÖ»ÓÐÒ»žö
	*/
	getPattern(frame(best_box),pEx,mean,stdev);

	//Get Fern features on warped patches
	Mat img;
	Mat warped;
	GaussianBlur(frame,img,Size(9,9),1.5);
	warped = img(bbhull);
	RNG& rng = theRNG();
	Point2f pt(bbhull.x+(bbhull.width-1)*0.5f,bbhull.y+(bbhull.height-1)*0.5f);
	vector<int> fern(classifier.getNumStructs());
	pX.clear();
	Mat patch;
	if (pX.capacity()<num_warps*good_boxes.size())
		pX.reserve(num_warps*good_boxes.size());
	int idx;
	for (int i=0;i<num_warps;i++)
	{
		/*
		2. ÊôÓÚPatchGeneratorÀàµÄ¹¹Ôìº¯Êý£¬ÓÃÀŽ¶ÔÍŒÏñÇøÓòœøÐÐ·ÂÉä±ä»»£¬ÏÈRNGÒ»žöËæ»úÒò×Ó£¬ÔÙµ÷ÓÃ£š£©ÔËËã·û²úÉúÒ»žö±ä»»ºóµÄÕýÑù±Ÿ¡£
		*/
		if (i>0)
			generator(frame,pt,warped,bbhull.size(),rng);
		for (int b=0; b<good_boxes.size(); b++)
		{
			idx=good_boxes[b];
			patch = img(*(grid[idx]));
			/*
			3.µÃµœÊäÈëµÄpatchµÄÌØÕ÷fern£š13Î»µÄ¶þœøÖÆŽúÂë£©
			*/
			classifier.getFeatures(patch,grid[idx]->sidx,fern);
			/*
			4.positive ferns <features, labels=1>È»ºó±êŒÇÎªÕýÑù±Ÿ£¬ŽæÈëpX£šÓÃÓÚŒ¯ºÏ·ÖÀàÆ÷µÄÕýÑù±Ÿ£©ÕýÑù±Ÿ¿â
			*/
			pX.push_back(make_pair(fern,1));
		}
	}
	printf("Positive examples generated: ferns:%d NN:1\n",(int)pX.size());
}

void TLD::getPattern(const Mat& img, Mat& pattern,Scalar& mean,Scalar& stdev)
{
	//Output: resized Zero-Mean patch
	resize(img,pattern,Size(patch_size,patch_size));
	meanStdDev(pattern,mean,stdev);
	pattern.convertTo(pattern,CV_32F);
	pattern = pattern-mean.val[0];
}

void TLD::generateNegativeData(const Mat& frame)
{
	/* Inputs:
	* - Image
	* - bad_boxes (Boxes far from the bounding box)
	* - variance (pEx variance)
	* Outputs
	* - Negative fern features (nX)
	* - Negative NN examples (nEx)
	*/
	random_shuffle(bad_boxes.begin(), bad_boxes.end());//Random shuffle bad_boxes indexes
	int idx;
	//Get Fern Features of the boxes with big variance (calculated using integral images)
	int a=0;
	//int num = std::min((int)bad_boxes.size(),(int)bad_patches*100); //limits the size of bad_boxes to try
	printf("negative data generation started.\n");
	vector<int> fern(classifier.getNumStructs());
	nX.reserve(bad_boxes.size());
	Mat patch;
	for (int j=0; j<bad_boxes.size(); j++)
	{
		idx = bad_boxes[j];
		if (getVar(*(grid[idx]),iisum,iisqsum)<var*0.5f)
			continue;
		patch =  frame(*(grid[idx]));
		classifier.getFeatures(patch,grid[idx]->sidx,fern);
		nX.push_back(make_pair(fern,0));
		a++;
	}
	printf("Negative examples generated: ferns: %d ",a);
	//random_shuffle(bad_boxes.begin(),bad_boxes.begin()+bad_patches);//Randomly selects 'bad_patches' and get the patterns for NN;
	Scalar dum1, dum2;
	nEx=vector<Mat>(bad_patches);
	for (int i=0;i<bad_patches;i++)
	{
		idx=bad_boxes[i];
		patch = frame(*(grid[idx]));
		/*
		getPatternº¯Êýœ«frameÍŒÏñbad_boxÇøÓòµÄÍŒÏñÆ¬¹éÒ»»¯µœ15*15ŽóÐ¡µÄpatch£¬ŽæÔÚnEx£šÓÃÓÚ×îœüÁÚ·ÖÀàÆ÷µÄžºÑù±Ÿ£©žºÑù±ŸÖÐ
		*/
		getPattern(patch,nEx[i],dum1,dum2);
	}
	printf("NN: %d\n",(int)nEx.size());
}

double TLD::getVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum){
	double brs = sum.at<int>(box.y+box.height,box.x+box.width);
	double bls = sum.at<int>(box.y+box.height,box.x);
	double trs = sum.at<int>(box.y,box.x+box.width);
	double tls = sum.at<int>(box.y,box.x);
	double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
	double blsq = sqsum.at<double>(box.y+box.height,box.x);
	double trsq = sqsum.at<double>(box.y,box.x+box.width);
	double tlsq = sqsum.at<double>(box.y,box.x);
	double mean = (brs+tls-trs-bls)/((double)box.area());
	double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
	return sqmean-mean*mean;
}

void TLD::processFrame(const cv::Mat& img1,const cv::Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2,BoundingBox& bbnext,
					   bool& lastboxfound, bool tl, FILE* bb_file)
{
	vector<BoundingBox> cbb;
	vector<float> cconf;
	int confident_detections=0;
	int didx; //detection index

	/*
	1. ÏÈœøÈëžú×ÙÄ£¿é
	*/
	if(lastboxfound && tl)
	{
		track(img1,img2,points1,points2);
	}
	else
	{
		tracked = false;
	}

	/*
	2. ÔÙœøÈëŒì²âÄ£¿é
	*/
	detect(img2);

	/*
	3. žú×ÙºÍŒì²âÈÚºÏ
	TLDÖ»žú×Ùµ¥Ä¿±ê£¬ËùÒÔ×ÛºÏÄ£¿é×ÛºÏžú×ÙÆ÷žú×ÙµœµÄµ¥žöÄ¿±êºÍŒì²âÆ÷¿ÉÄÜŒì²âµœµÄ¶àžöÄ¿±ê£¬È»ºóÖ»Êä³ö±£ÊØÏàËÆ¶È×îŽóµÄÒ»žöÄ¿±ê
	*/
	if (tracked) // Èç¹ûœøÐÐ¹ýžú×Ù
	{
		bbnext=tbb;
		lastconf=tconf;
		lastvalid=tvalid;
		printf("Tracked\n");
		if(detected)  // Èç¹ûœøÐÐ¹ýŒì²â
		{
			// 3.1 ÏÈÍš¹ý ÖØµþ¶È ¶ÔŒì²âÆ÷Œì²âµœµÄÄ¿±êbounding boxœøÐÐŸÛÀà£¬Ã¿žöÀàµÄÖØµþ¶ÈÐ¡ÓÚ0.5
			clusterConf(dbb,dconf,cbb,cconf);                       //   cluster detections
			printf("Found %d clusters\n",(int)cbb.size());
			// 3.2 ÔÙÕÒµœÓëžú×ÙÆ÷žú×ÙµœµÄboxŸàÀë±ÈœÏÔ¶µÄÀà£šŒì²âÆ÷Œì²âµœµÄbox£©£¬¶øÇÒËüµÄÏà¹ØÏàËÆ¶È±Èžú×ÙÆ÷µÄÒªŽó£ºŒÇÂŒÂú×ãÉÏÊöÌõŒþ£¬Ò²ŸÍÊÇ¿ÉÐÅ¶È±ÈœÏžßµÄÄ¿±êboxµÄžöÊý
			for (int i=0;i<cbb.size();i++)
			{
				if (bbOverlap(tbb,cbb[i])<0.5 && cconf[i]>tconf)
				{  //  Get index of a clusters that is far from tracker and are more confident than the tracker
					confident_detections++;
					didx=i; //detection index
				}
			}
			// 3.2.1 ÅÐ¶ÏÈç¹ûÖ»ÓÐÒ»žöÂú×ãÉÏÊöÌõŒþµÄbox£¬ÄÇÃŽŸÍÓÃÕâžöÄ¿±êboxÀŽÖØÐÂ³õÊŒ»¯žú×ÙÆ÷£šÒ²ŸÍÊÇÓÃŒì²âÆ÷µÄœá¹ûÈ¥ŸÀÕýžú×ÙÆ÷
			if (confident_detections==1)
			{                                //if there is ONE such a cluster, re-initialize the tracker
				printf("Found a better match..reinitializing tracking\n");
				bbnext=cbb[didx];
				lastconf=cconf[didx];
				lastvalid=false;
			}
			// 3.2.2 Èç¹ûÂú×ãÉÏÊöÌõŒþµÄbox²»Ö»Ò»žö£¬ÄÇÃŽŸÍÕÒµœŒì²âÆ÷Œì²âµœµÄboxÓëžú×ÙÆ÷Ô€²âµœµÄboxŸàÀëºÜœü£šÖØµþ¶ÈŽóÓÚ0.7£©µÄËùÒÔbox£¬¶ÔÆä×ø±êºÍŽóÐ¡œøÐÐÀÛŒÓ
			else
			{
				printf("%d confident cluster was found\n",confident_detections);
				int cx=0,cy=0,cw=0,ch=0;
				int close_detections=0;
				for (int i=0; i<dbb.size(); i++)
				{
					if(bbOverlap(tbb,dbb[i])>0.7)
					{                     // Get mean of close detections
						cx += dbb[i].x;
						cy +=dbb[i].y;
						cw += dbb[i].width;
						ch += dbb[i].height;
						close_detections++;
						printf("weighted detection: %d %d %d %d\n",dbb[i].x,dbb[i].y,dbb[i].width,dbb[i].height);
					}
				}
				if (close_detections>0)
				{
					bbnext.x = cvRound((float)(10*tbb.x+cx)/(float)(10+close_detections));   // weighted average trackers trajectory with the close detections
					bbnext.y = cvRound((float)(10*tbb.y+cy)/(float)(10+close_detections));
					bbnext.width = cvRound((float)(10*tbb.width+cw)/(float)(10+close_detections));
					bbnext.height =  cvRound((float)(10*tbb.height+ch)/(float)(10+close_detections));
					printf("Tracker bb: %d %d %d %d\n",tbb.x,tbb.y,tbb.width,tbb.height);
					printf("Average bb: %d %d %d %d\n",bbnext.x,bbnext.y,bbnext.width,bbnext.height);
					printf("Weighting %d close detection(s) with tracker..\n",close_detections);
				}
				else
				{
					printf("%d close detections were found\n",close_detections);

				}
			}
		}
	}
	else // Èç¹ûžú×ÙÊ§°Ü£¬Ö»ÓÐŒì²âœá¹û
	{                                       //   If NOT tracking
		printf("Not tracking..\n");
		lastboxfound = false;
		lastvalid = false;
		if(detected)
		{                           //  and detector is defined
			clusterConf(dbb,dconf,cbb,cconf);   //  cluster detections
			printf("Found %d clusters\n",(int)cbb.size());
			if (cconf.size()==1)
			{
				bbnext=cbb[0]; // Ö»°ÑŸÛÀàœá¹ûµÄµÚÒ»žöÀà×÷Îª×îÖÕœá¹û
				lastconf=cconf[0];
				printf("Confident detection..reinitializing tracker\n");
				lastboxfound = true;
			}
		}
	}
	lastbox=bbnext;
	if (lastboxfound)
		fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
	else
		fprintf(bb_file,"NaN,NaN,NaN,NaN,NaN\n");

	/*
	4. Ñ§Ï°Ä£¿é
	*/
	if (lastvalid && tl)
		learn(img2);
}


void TLD::track(const Mat& img1, const Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2){
	/*Inputs:
	* -current frame(img2), last frame(img1), last Bbox(bbox_f[0]).
	*Outputs:
	*- Confidence(tconf), Predicted bounding box(tbb),Validity(tvalid), points2 (for display purposes only)
	*/
	/*
	ÏÈÔÚlastboxÖÐŸùÔÈ²ÉÑù10*10=100žöÌØÕ÷µã£šÍøžñŸùÔÈÈöµã£©£¬ŽæÓÚpoints1
	*/
	bbPoints(points1,lastbox);

	if (points1.size()<1)
	{
		printf("BB= %d %d %d %d, Points not generated\n",lastbox.x,lastbox.y,lastbox.width,lastbox.height);
		tvalid=false;
		tracked=false;
		return;
	}
	vector<Point2f> points = points1;

	/*
	TLDžú×ÙÄ£¿éµÄÊµÏÖÊÇÀûÓÃÁËMedia Flow ÖÐÖµ¹âÁ÷žú×ÙºÍžú×ÙŽíÎóŒì²âËã·šµÄœáºÏ¡£ÖÐÖµÁ÷žú×Ù·œ·šÊÇ»ùÓÚForward-Backward ErrorºÍNNCµÄ¡£
	Ô­ÀíºÜŒòµ¥£ºŽÓtÊ±¿ÌµÄÍŒÏñµÄAµã£¬žú×Ùµœt+1Ê±¿ÌµÄÍŒÏñBµã£»È»ºóµ¹»ØÀŽ£¬ŽÓt+1Ê±¿ÌµÄÍŒÏñµÄBµãÍù»Øžú×Ù£¬ŒÙÈçžú×ÙµœtÊ±¿ÌµÄÍŒÏñµÄCµã£¬
	ÕâÑùŸÍ²úÉúÁËÇ°ÏòºÍºóÏòÁœžö¹ìŒ££¬±ÈœÏtÊ±¿ÌÖÐ AµãºÍCµãµÄŸàÀë£¬Èç¹ûŸàÀëÐ¡ÓÚÒ»žöãÐÖµ£¬ÄÇÃŽŸÍÈÏÎªÇ°Ïòžú×ÙÊÇÕýÈ·µÄ£»ÕâžöŸàÀëŸÍÊÇFB_error
	*/
	tracked = tracker.trackf2f(img1,img2,points,points2);

	if (tracked)
	{
		//points2ËùŸÛŒ¯µÄµãµÄÍâºÐÔ€²â
		/*
		Ô€²âbounding boxÔÚµ±Ç°Ö¡µÄÎ»ÖÃºÍŽóÐ¡tbb
		*/
		bbPredict(points,points2,lastbox,tbb);

		/*
		žú×ÙÊ§°ÜŒì²â£ºÈç¹ûFB errorµÄÖÐÖµŽóÓÚ10žöÏñËØ£šŸ­ÑéÖµ£©£¬»òÕßÔ€²âµœµÄµ±Ç°boxµÄÎ»ÖÃÒÆ³öÍŒÏñ£¬ÔòÈÏÎªžú×ÙŽíÎó
		*/
		if (tracker.getFB()>10 || tbb.x>img2.cols ||  tbb.y>img2.rows || tbb.br().x < 1 || tbb.br().y <1)
		{
			tvalid =false; //too unstable prediction or bounding box out of image
			tracked = false;
			printf("Too unstable predictions FB error=%f\n",tracker.getFB());
			return;
		}

		//Estimate Confidence and Validity
		Mat pattern;
		Scalar mean, stdev;
		BoundingBox bb;
		bb.x = max(tbb.x,0);
		bb.y = max(tbb.y,0);
		bb.width = min(min(img2.cols-tbb.x,tbb.width),min(tbb.width,tbb.br().x));
		bb.height = min(min(img2.rows-tbb.y,tbb.height),min(tbb.height,tbb.br().y));
		/*
		¹éÒ»»¯img2(bb)¶ÔÓŠµÄpatchµÄsize£š·ÅËõÖÁpatch_size = 15*15£©£¬ŽæÈëpattern
		*/
		getPattern(img2(bb),pattern,mean,stdev);

		vector<int> isin;
		float dummy;
		/*
		ŒÆËãÍŒÏñÆ¬patternµœÔÚÏßÄ£ÐÍMµÄ±£ÊØÏàËÆ¶È
		*/
		classifier.NNConf(pattern,isin,dummy,tconf); //Conservative Similarity

		/*
		Èç¹û±£ÊØÏàËÆ¶ÈŽóÓÚãÐÖµ£¬ÔòÆÀ¹À±ŸŽÎžú×ÙÓÐÐ§£¬·ñÔòžú×ÙÎÞÐ§
		*/
		tvalid = lastvalid;
		if (tconf>classifier.thr_nn_valid)
		{
			tvalid =true;
		}
	}
	else
		printf("No points tracked\n");

}

void TLD::bbPoints(vector<cv::Point2f>& points,const BoundingBox& bb)
{
	int max_pts=10;
	int margin_h=0;
	int margin_v=0;
	int stepx = ceil((double)(bb.width-2*margin_h)/max_pts);
	int stepy = ceil((double)(bb.height-2*margin_v)/max_pts);
	for (int y=bb.y+margin_v;y<bb.y+bb.height-margin_v;y+=stepy)
	{
		for (int x=bb.x+margin_h;x<bb.x+bb.width-margin_h;x+=stepx)
		{
			points.push_back(Point2f(x,y));
		}
	}
}

void TLD::bbPredict(const vector<cv::Point2f>& points1,const vector<cv::Point2f>& points2,
					const BoundingBox& bb1,BoundingBox& bb2)
{
	int npoints = (int)points1.size();
	vector<float> xoff(npoints);
	vector<float> yoff(npoints);
	printf("tracked points : %d\n",npoints);
	for (int i=0;i<npoints;i++)
	{
		xoff[i]=points2[i].x-points1[i].x;
		yoff[i]=points2[i].y-points1[i].y;
	}
	float dx = median(xoff);
	float dy = median(yoff);
	float s; // sÊÇžöËõ·ÅÒò×Ó
	if (npoints>1)
	{
		vector<float> d;
		d.reserve(npoints*(npoints-1)/2);
		for (int i=0;i<npoints;i++)
		{
			for (int j=i+1;j<npoints;j++)
			{
				d.push_back(norm(points2[i]-points2[j])/norm(points1[i]-points1[j]));
			}
		}
		s = median(d);
	}
	else
	{
		s = 1.0;
	}
	float s1 = 0.5*(s-1)*bb1.width;
	float s2 = 0.5*(s-1)*bb1.height;
	printf("s= %f s1= %f s2= %f \n",s,s1,s2);
	bb2.x = cvRound( bb1.x + dx -s1);
	bb2.y = cvRound( bb1.y + dy -s2);
	bb2.width = cvRound(bb1.width*s);
	bb2.height = cvRound(bb1.height*s);
	printf("predicted bb: %d %d %d %d\n",bb2.x,bb2.y,bb2.br().x,bb2.br().y);
}

void TLD::detect(const cv::Mat& frame)
{
	//cleaning
	dbb.clear();
	dconf.clear();
	dt.bb.clear();
	double t = (double)getTickCount();
	Mat img(frame.rows,frame.cols,CV_8U);

	/*
	1. ·œ²î·ÖÀàÆ÷
	*/
	// 1.1 ŒÆËã·œ²î»ý·ÖÍŒ
	integral(frame,iisum,iisqsum);
	// 1.2 žßË¹Ä£ºý£¬È¥Ôë
	GaussianBlur(frame,img,Size(9,9),1.5);

	/*
	2. Œ¯ºÏ·ÖÀàÆ÷
	*/
	// 2.1 Œ¯ºÏ·ÖÀàÆ÷±äÁ¿³õÊŒ»¯
	int numtrees = classifier.getNumStructs();
	float fern_th = classifier.getFernTh();
	vector <int> ferns(10);
	float conf;
	int a=0;
	Mat patch;
	// ·œ²î·ÖÀàÆ÷ºÍŒ¯ºÏ·ÖÀàÆ÷·ÅÒ»ÆðÁË
	/*
	Œ¯ºÏ·ÖÀàÆ÷£šËæ»úÉ­ÁÖ£©¹²ÓÐ10¿ÅÊ÷£š»ù±Ÿ·ÖÀàÆ÷£©£¬Ã¿¿ÃÊ÷13žöÅÐ¶ÏœÚµã£¬Ã¿žöÅÐ¶ÏœÚµãŸ­±ÈœÏµÃµœÒ»žö¶þœøÖÆÎ»0»òÕß1£¬
	ÕâÑùÃ¿¿ÃÊ÷ŸÍ¶ÔÓŠµÃµœÒ»žö13Î»µÄ¶þœøÖÆÂëx£šÒ¶×Ó£©£¬Õâžö¶þœøÖÆÂëx¶ÔÓŠÓÚÒ»žöºóÑéžÅÂÊP(y|x)¡£ÄÇÃŽÕûÒ»žöŒ¯ºÏ·ÖÀàÆ÷£š¹²10žö»ù±Ÿ·ÖÀàÆ÷£©ŸÍÓÐ10žöºóÑéžÅÂÊÁË£¬
	œ«10žöºóÑéžÅÂÊœøÐÐÆœŸù£¬Èç¹ûŽóÓÚãÐÖµ£šÒ»¿ªÊŒÉèŸ­ÑéÖµ0.65£¬ºóÃæÔÙÑµÁ·ÓÅ»¯£©µÄ»°£¬ŸÍÈÏÎªžÃÍŒÏñÆ¬º¬ÓÐÇ°Ÿ°Ä¿±ê
	*/
	for (int i=0; i<grid.size(); i++)
	{//FIXME: BottleNeck
		// 1.3 ÀûÓÃ»ý·ÖÍŒŒÆËãÃ¿žöŽýŒì²âŽ°¿ÚµÄ·œ²î£¬·œ²îŽóÓÚvarãÐÖµ£šÄ¿±êpatch·œ²îµÄ50%£©µÄ£¬ÔòÈÏÎªÆäº¬ÓÐÇ°Ÿ°Ä¿±ê£¬Íš¹ýžÃÄ£¿éµÄœøÈëŒ¯ºÏ·ÖÀàÆ÷Ä£¿é
		if (getVar(*(grid[i]),iisum,iisqsum)>=var)
		{
			a++;
			patch = img(*(grid[i]));// ¿ÉÒÔÈ¡³öÍŒÏñÖÐgrid[i]ŽóÐ¡µÄÒ»Æ¬×ÓÍŒÏñ
			// 2.2 ÏÈµÃµœžÃpatchµÄÌØÕ÷Öµ£š13Î»µÄ¶þœøÖÆŽúÂë£©
			classifier.getFeatures(patch,grid[i]->sidx,ferns);
			// 2.3 ÔÙŒÆËãžÃÌØÕ÷Öµ¶ÔÓŠµÄºóÑéžÅÂÊÀÛŒÓÖµ
			conf = classifier.measure_forest(ferns);
			tmp.conf[i]=conf;
			tmp.patt[i]=ferns;
			// 2.4 ÈôŒ¯ºÏ·ÖÀàÆ÷µÄºóÑéžÅÂÊµÄÆœŸùÖµŽóÓÚãÐÖµfern_th£šÓÉÑµÁ·µÃµœ£©£¬ŸÍÈÏÎªº¬ÓÐÇ°Ÿ°Ä¿±ê
			if (conf>numtrees*fern_th)
			{
				//œ«Íš¹ýÒÔÉÏÁœžöŒì²âÄ£¿éµÄÉšÃèŽ°¿ÚŒÇÂŒÔÚdetect structureÖÐ
				dt.bb.push_back(i);
			}
		}
		else
			tmp.conf[i]=0.0;
	}
	int detections = dt.bb.size();
	printf("%d Bounding boxes passed the variance filter\n",a);
	printf("%d Initial detection from Fern Classifier\n",detections);
	//Èç¹ûË³ÀûÍš¹ýÒÔÉÏÁœžöŒì²âÄ£¿éµÄÉšÃèŽ°¿ÚÊýŽóÓÚ100žö£¬ÔòÖ»È¡ºóÑéžÅÂÊŽóµÄÇ°100žö£¬×öÅÅÐòÁË
	if (detections>100)
	{
		nth_element(dt.bb.begin(),dt.bb.begin()+100,dt.bb.end(),CComparator(tmp.conf));
		dt.bb.resize(100);
		detections=100;
	}
	//  for (int i=0;i<detections;i++){
	//        drawBox(img,grid[dt.bb[i]]);
	//    }
	//  imshow("detections",img);
	if (detections==0){
		detected=false;
		return;
	}
	printf("Fern detector made %d detections ",detections);
	t=(double)getTickCount()-t;
	printf("in %gms\n", t*1000/getTickFrequency());
	//  ³õÊŒ»¯Œì²âœá¹¹Ìå
	dt.patt = vector<vector<int> >(detections,vector<int>(10,0));        //  Corresponding codes of the Ensemble Classifier
	dt.conf1 = vector<float>(detections);                                //  Relative Similarity (for final nearest neighbour classifier)
	dt.conf2 =vector<float>(detections);                                 //  Conservative Similarity (for integration with tracker)
	dt.isin = vector<vector<int> >(detections,vector<int>(3,-1));        //  Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
	dt.patch = vector<Mat>(detections,Mat(patch_size,patch_size,CV_32F));//  Corresponding patches
	int idx;
	Scalar mean, stdev;

	/*
	3. ×îœüÁÚ·ÖÀàÆ÷
	*/
	float nn_th = classifier.getNNTh();
	for (int i=0; i<detections; i++)	//  for every remaining detection
	{
		idx=dt.bb[i];                                            //  Get the detected bounding box index
		patch = frame(*(grid[idx]));
		// 3.1 ÏÈ¹éÒ»»¯patchµÄsize£š·ÅËõÖÁpatch_size = 15*15£©£¬ŽæÈëdt.patch[i]
		getPattern(patch,dt.patch[i],mean,stdev);                //  Get pattern within bounding box
		// 3.2 ŒÆËãÍŒÏñÆ¬patternµœÔÚÏßÄ£ÐÍMµÄÏà¹ØÏàËÆ¶ÈºÍ±£ÊØÏàËÆ¶È
		classifier.NNConf(dt.patch[i],dt.isin[i],dt.conf1[i],dt.conf2[i]);  //  Evaluate nearest neighbour classifier
		dt.patt[i]=tmp.patt[idx];
		//printf("Testing feature %d, conf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);
		if (dt.conf1[i]>nn_th)
		{                                               //  idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour
			dbb.push_back(*(grid[idx]));                                         //  BB    = dt.bb(:,idx); % bounding boxes
			dconf.push_back(dt.conf2[i]);                                     //  Conf  = dt.conf2(:,idx); % conservative confidences
		}
	}                                                                         //  end

	/*
	µœÄ¿Ç°ÎªÖ¹£¬Œì²âÆ÷Œì²âÍê³É£¬È«²¿Íš¹ýÈýžöŒì²âÄ£¿éµÄÉšÃèŽ°¿ÚŽæÔÚdbbÖÐ
	*/
	if (dbb.size()>0){
		printf("Found %d NN matches\n",(int)dbb.size());
		detected=true;
	}
	else{
		printf("No NN matches found.\n");
		detected=false;
	}
}

void TLD::evaluate(){
}

void TLD::learn(const Mat& img)
{
	printf("[Learning] ");
	///Check consistency
	BoundingBox bb;
	bb.x = max(lastbox.x,0);
	bb.y = max(lastbox.y,0);
	bb.width = min(min(img.cols-lastbox.x,lastbox.width),min(lastbox.width,lastbox.br().x));
	bb.height = min(min(img.rows-lastbox.y,lastbox.height),min(lastbox.height,lastbox.br().y));
	Scalar mean, stdev;
	Mat pattern;
	getPattern(img(bb),pattern,mean,stdev);
	vector<int> isin;
	float dummy, conf;
	classifier.NNConf(pattern,isin,conf,dummy);
	if (conf<0.5) {
		printf("Fast change..not training\n");
		lastvalid =false;
		return;
	}
	if (pow(stdev.val[0],2)<var){
		printf("Low variance..not training\n");
		lastvalid=false;
		return;
	}
	if(isin[2]==1){
		printf("Patch in negative data..not traing");
		lastvalid=false;
		return;
	}
	/// Data generation
	for (int i=0;i<grid.size();i++){
		grid[i]->overlap = bbOverlap(lastbox,*(grid[i]));
	}
	vector<pair<vector<int>,int> > fern_examples;
	good_boxes.clear();
	bad_boxes.clear();
	getOverlappingBoxes(lastbox,num_closest_update);
	if (good_boxes.size()>0)
		generatePositiveData(img,num_warps_update);
	else{
		lastvalid = false;
		printf("No good boxes..Not training");
		return;
	}
	fern_examples.reserve(pX.size()+bad_boxes.size());
	fern_examples.assign(pX.begin(),pX.end());
	int idx;
	for (int i=0;i<bad_boxes.size();i++){
		idx=bad_boxes[i];
		if (tmp.conf[idx]>=1){
			fern_examples.push_back(make_pair(tmp.patt[idx],0));
		}
	}
	vector<Mat> nn_examples;
	nn_examples.reserve(dt.bb.size()+1);
	nn_examples.push_back(pEx);
	for (int i=0;i<dt.bb.size();i++){
		idx = dt.bb[i];
		if (bbOverlap(lastbox,*(grid[idx])) < bad_overlap)
			nn_examples.push_back(dt.patch[i]);
	}
	/// Classifiers update
	classifier.trainF(fern_examples,2);
	classifier.trainNN(nn_examples);
	classifier.show();
}

void TLD::buildGrid(const cv::Mat& img, const cv::Rect& box)
{
	const float SHIFT = 0.15;
	const float SCALES[] = {0.16151,0.19381,0.23257,0.27908,0.33490,0.40188,0.48225,
		0.57870,0.69444,0.83333,1,1.20000,1.44000,1.72800,
		2.07360,2.48832,2.98598,3.58318,4.29982,5.15978,6.19174};
	int width, height, min_bb_side;
	//Rect bbox;
	BoundingBox* pbbox;
	Size scale;
	int sc=0;
	for (int s=0;s<21;s++)
	{
		width = cvRound(box.width*SCALES[s]);
		height = cvRound(box.height*SCALES[s]);
		min_bb_side = min(height,width);
		if (min_bb_side < min_win || width > img.cols || height > img.rows)
			continue;
		scale.width = width;
		scale.height = height;
		scales.push_back(scale);
		for (int y = 1; y < img.rows-height; y += cvRound(SHIFT*min_bb_side))//cvRound to int
		{
			for (int x=1; x < img.cols-width; x += cvRound(SHIFT*min_bb_side))//cvRound to int
			{
				pbbox = new BoundingBox;
				pbbox->x = x;
				pbbox->y = y;
				pbbox->width = width;
				pbbox->height = height;
				pbbox->overlap = bbOverlap(*pbbox,BoundingBox(box));
				pbbox->sidx = sc;
				grid.push_back(pbbox);
			}
		}
		sc++;
	}
}

float TLD::bbOverlap(const BoundingBox& box1,const BoundingBox& box2)
{
	if (box1.x > box2.x+box2.width) { return 0.0; }
	if (box1.y > box2.y+box2.height) { return 0.0; }
	if (box1.x+box1.width < box2.x) { return 0.0; }
	if (box1.y+box1.height < box2.y) { return 0.0; }

	float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
	float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

	float intersection = colInt * rowInt;
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return intersection / (area1 + area2 - intersection);
}

void TLD::getOverlappingBoxes(const cv::Rect& box1,int num_closest)
{
	float max_overlap = 0;
	for (int i=0;i<grid.size();i++)
	{
		if (grid[i]->overlap > max_overlap)
		{
			max_overlap = grid[i]->overlap;
			best_box = *(grid[i]);
		}
		if (grid[i]->overlap > 0.6)
		{
			good_boxes.push_back(i);
		}
		else if (grid[i]->overlap < bad_overlap)
		{
			bad_boxes.push_back(i);
		}
	}
	//Get the best num_closest (10) boxes and puts them in good_boxes
	if (good_boxes.size()>num_closest)
	{
		std::nth_element(good_boxes.begin(),good_boxes.begin()+num_closest,good_boxes.end(),OComparator(grid));
		good_boxes.resize(num_closest);
	}
	getBBHull();
}

void TLD::getBBHull()
{
	int x1=INT_MAX, x2=0;
	int y1=INT_MAX, y2=0;
	int idx;
	for (int i=0;i<good_boxes.size();i++)
	{
		idx= good_boxes[i];
		x1=min(grid[idx]->x,x1);
		y1=min(grid[idx]->y,y1);
		x2=max(grid[idx]->x+grid[idx]->width,x2);
		y2=max(grid[idx]->y+grid[idx]->height,y2);
	}
	bbhull.x = x1;
	bbhull.y = y1;
	bbhull.width = x2-x1;
	bbhull.height = y2 -y1;
}

bool bbcomp(const BoundingBox& b1,const BoundingBox& b2)
{
	TLD t;
	if (t.bbOverlap(b1,b2)<0.5)
		return false;
	else
		return true;
}

int TLD::clusterBB(const vector<BoundingBox>& dbb,vector<int>& indexes)
{
	//FIXME: Conditional jump or move depends on uninitialised value(s)
	const int c = dbb.size();
	//1. Build proximity matrix
	Mat D(c,c,CV_32F);
	float d;
	for (int i=0;i<c;i++){
		for (int j=i+1;j<c;j++){
			d = 1-bbOverlap(dbb[i],dbb[j]);
			D.at<float>(i,j) = d;
			D.at<float>(j,i) = d;
		}
	}
	//2. Initialize disjoint clustering
	/*float L[c-1];
	int nodes[c-1][2];
	int belongs[c];*/

	float *L=new float[c-1]; //Level
	int **nodes=new int*[c];
	for (int i=0;i<c;i++)
	{
		nodes[i]=new int[2];
	}

	int * belongs=new int[c];

	int m=c;
	for (int i=0;i<c;i++){
		belongs[i]=i;
	}
	for (int it=0;it<c-1;it++){
		//3. Find nearest neighbor
		float min_d = 1;
		int node_a, node_b;
		for (int i=0;i<D.rows;i++){
			for (int j=i+1;j<D.cols;j++){
				if (D.at<float>(i,j)<min_d && belongs[i]!=belongs[j]){
					min_d = D.at<float>(i,j);
					node_a = i;
					node_b = j;
				}
			}
		}
		if (min_d>0.5){
			int max_idx =0;
			bool visited;
			for (int j=0;j<c;j++){
				visited = false;
				for(int i=0;i<2*c-1;i++){
					if (belongs[j]==i){
						indexes[j]=max_idx;
						visited = true;
					}
				}
				if (visited)
					max_idx++;
			}
			return max_idx;
		}

		//4. Merge clusters and assign level
		L[m]=min_d;
		nodes[it][0] = belongs[node_a];
		nodes[it][1] = belongs[node_b];
		for (int k=0;k<c;k++){
			if (belongs[k]==belongs[node_a] || belongs[k]==belongs[node_b])
				belongs[k]=m;
		}
		m++;
	}
	//delete L;

	delete[]nodes;
	delete[]L;
	delete[] belongs;
	nodes=NULL;
	L=NULL;
	belongs=NULL;
	return 1;

}

void TLD::clusterConf(const vector<BoundingBox>& dbb,const vector<float>& dconf,vector<BoundingBox>& cbb,vector<float>& cconf)
{
	int numbb = dbb.size();
	vector<int> T;
	float space_thr = 0.5;
	int c=1;
	switch (numbb)
	{
	case 1:
		cbb=vector<BoundingBox>(1,dbb[0]);
		cconf=vector<float>(1,dconf[0]);
		return;
		break;
	case 2:
		T =vector<int>(2,0);
		if (1-bbOverlap(dbb[0],dbb[1])>space_thr)
		{
			T[1]=1;
			c=2;
		}
		break;
	default:
		T = vector<int>(numbb,0);
		c = partition(dbb,T,(*bbcomp));
		//c = clusterBB(dbb,T);
		break;
	}
	cconf=vector<float>(c);
	cbb=vector<BoundingBox>(c);
	printf("Cluster indexes: ");
	BoundingBox bx;
	for (int i=0;i<c;i++)
	{
		float cnf=0;
		int N=0,mx=0,my=0,mw=0,mh=0;
		for (int j=0;j<T.size();j++)
		{
			if (T[j]==i)
			{
				printf("%d ",i);
				cnf=cnf+dconf[j];
				mx=mx+dbb[j].x;
				my=my+dbb[j].y;
				mw=mw+dbb[j].width;
				mh=mh+dbb[j].height;
				N++;
			}
		}
		if (N>0)
		{
			cconf[i]=cnf/N;
			bx.x=cvRound(mx/N);
			bx.y=cvRound(my/N);
			bx.width=cvRound(mw/N);
			bx.height=cvRound(mh/N);
			cbb[i]=bx;
		}
	}
	printf("\n");
}

