#include <XnCppWrapper.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <fstream>
#include <iostream>

xn::Context g_Context;
xn::ScriptNode g_scriptNode;
xn::UserGenerator g_UserGenerator;
xn::DepthGenerator g_DepthGenerator;

using namespace xn;
using namespace cv;
using namespace std;



#define MAX_DEPTH 10000
float g_pDepthHist[MAX_DEPTH];
XnRGB24Pixel* g_pTexMap = NULL;
unsigned int g_nTexMapX = 0;
unsigned int g_nTexMapY = 0;

XnBool g_bNeedPose = FALSE;
XnChar g_strPose[20] = "";

string relation_name = "gesture";

#define MAX_NUM_USERS 15
//---------------------------------------------------------------------------
// Code
//---------------------------------------------------------------------------

bool isFileEmpty(fstream& file){
    file.seekg(0, ios::end);
    return file.tellg() == 0;
}

typedef struct Frame{
    public:
        int rightHandX, rightHandY, rightHandZ, handConfigurationRight;
        int leftHandX, leftHandY, leftHandZ, handConfigurationLeft;
        int torsoX, torsoY, torsoZ;
        int headX, headY;
        int htRatioX, htRatioY;
        float rightVectorX, rightVectorY, rightVectorZ;
        float leftVectorX, leftVectorY, leftVectorZ;

    
    Frame(int rhX, int rhY, int rhZ, int hc1, int lhX, int lhY, int lhZ, int hc2, int tX, int tY, int tZ, int hX, int hY){
        rightHandX = rhX;
        rightHandY = rhY;
        rightHandZ = rhZ;
        handConfigurationRight = hc1;

        leftHandX = lhX;
        leftHandY = lhY;
        leftHandZ = lhZ;
        handConfigurationLeft = hc2;


        torsoX = tX;
        torsoY = tY;
        torsoZ = tZ;

        headX = hX;
        headY = hY;

        htRatioX = sqrt(pow(torsoX - headX, 2));
        htRatioY = sqrt(pow(torsoY - headY, 2));
        

        rightVectorX = sqrt(pow(rightHandX - torsoX, 2)); //Para não deixar o valor negativo
        rightVectorY = sqrt(pow(rightHandY - torsoY, 2)); //Para não deixar o valor negativo
        rightVectorZ = sqrt(pow(rightHandZ - torsoZ, 2)); //Para não deixar o valor negativo

        leftVectorX = sqrt(pow(leftHandX - torsoX, 2)); //Para não deixar o valor negativo
        leftVectorY = sqrt(pow(leftHandY - torsoY, 2)); //Para não deixar o valor negativo
        leftVectorZ = sqrt(pow(leftHandZ - torsoZ, 2)); //Para não deixar o valor negativo

        int htRatio = sqrt(pow(htRatioX, 2) + pow(htRatioY, 2)); //Magnitude do Ratio da Cabeça e Tronco

        rightVectorX /= htRatio;
        rightVectorY /= htRatio;


        leftVectorX /= htRatio;
        leftVectorY /= htRatio;
    }

    void WriteToFile(fstream& file){
        if(isFileEmpty(file))
            WriteArff(file);
        file << rightVectorX << "\t" << rightVectorY << "\t" << rightVectorZ << "\t" << handConfigurationRight << "\t"; 
        file << leftVectorX << "\t" << leftVectorY << "\t" << leftVectorZ << "\t" << handConfigurationLeft << endl;
    }

    void WriteArff(fstream & file){
        file << "@RELATION " << relation_name << endl;
        file << "@ATTRIBUTE rightVectorX numeric" << endl << "@ATTRIBUTE rightVectorY numeric" << endl << "@ATTRIBUTE rightVectorZ numeric" << endl << "@ATTRIBUTE handConfigurationRight numeric" << endl; 
        file << "@ATTRIBUTE leftVectorX numeric" << endl << "@ATTRIBUTE leftVectorY numeric" << endl << "@ATTRIBUTE leftVectorZ numeric" << endl <<"@ATTRIBUTE handConfigurationLeft numeric" << endl;
        file << "@DATA" << endl;
    }

}Frame;


XnBool fileExists(const char *fn)
{
	XnBool exists;
	xnOSDoesFileExist(fn, &exists);
	return exists;
}

// Callback:User has exited the scene.Wait for 10sec to reEnter,or declare user Lost
void XN_CALLBACK_TYPE UserExit(xn::UserGenerator & generate,XnUserID nID,void* pCookie)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d User Exited\n", epochTime);
}

// Callback: New has reEntered the Scene

void XN_CALLBACK_TYPE UserReEnter(xn::UserGenerator & generate,XnUserID nID,void* pCookie)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d User ReEntered\n", epochTime);
}

//Callback: New User has Enterd the Scene
void XN_CALLBACK_TYPE User_NewUser(xn::UserGenerator& generator, XnUserID nId, void* pCookie)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d New User %d\n", epochTime, nId);
    // New user found
    if (g_bNeedPose)
    {
        //If pose detection is required,Start detecting g_strPose pose.Once that is detected,
        // UserPose_PoseDetected callback will run and start callibration of user.
        g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
    }
    else
    {
        // If pose is not required, start callibration directly
        g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
    }
}
// Callback: An existing user was lost
void XN_CALLBACK_TYPE User_LostUser(xn::UserGenerator& generator, XnUserID nId, void* pCookie)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Lost user %d\n", epochTime, nId);
}
// Callback: Detected a pose
void XN_CALLBACK_TYPE UserPose_PoseDetected(xn::PoseDetectionCapability& capability, const XnChar* strPose, XnUserID nId, void* pCookie)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Pose %s detected for user %d\n", epochTime, strPose, nId);
    // When pose is detected,stop looking for the pose and start callibration
    g_UserGenerator.GetPoseDetectionCap().StopPoseDetection(nId);
    g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
}
// Callback: Started calibration
void XN_CALLBACK_TYPE UserCalibration_CalibrationStart(xn::SkeletonCapability& capability, XnUserID nId, void* pCookie)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Calibration started for user %d\n", epochTime, nId);
}

void XN_CALLBACK_TYPE UserCalibration_CalibrationComplete(xn::SkeletonCapability& capability, XnUserID nId, XnCalibrationStatus eStatus, void* pCookie)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    if (eStatus == XN_CALIBRATION_STATUS_OK)
    {
        // Calibration succeeded
        printf("%d Calibration complete, start tracking user %d\n", epochTime, nId);
        g_UserGenerator.GetSkeletonCap().StartTracking(nId);
    }
    else
    {
        // Calibration failed
        printf("%d Calibration failed for user %d\n", epochTime, nId);
        if(eStatus==XN_CALIBRATION_STATUS_MANUAL_ABORT)
        {
            printf("Manual abort occured, stop attempting to calibrate!");
            return;
        }
        if (g_bNeedPose)
        {
            g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
        }
        else
        {
            g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
        }
    }
}


#define CHECK_RC(nRetVal, what)					    \
    if (nRetVal != XN_STATUS_OK)				    \
{								    \
    printf("%s failed: %s\n", what, xnGetStatusString(nRetVal));    \
    return nRetVal;						    \
}

int main(int argc, char **argv)
{
    if(argc < 2){
        cout << "Erro (Uso: " << argv[0] << " <oni file>)" << endl;
        return -1;
    }


    cout << "+-----------Depth DataSet Generator-----------+" << endl; 
	cout << "|     Copyright Murilo Kinape Rivabem 2017    |" << endl;
	cout << "|                                             |" << endl;
	cout << "|	     Press Esc to exit the program         |" << endl;
	cout << "+---------------------------------------------+" << endl;



    string filename = "./Dataset/data.arff";
    std::fstream dataset(filename.c_str(), std::fstream::out | std::fstream::app);

    cv::Mat colorArr[3];
    cv::Mat colorImg;               // OpenCV cv::Mat structures
    
    cv::Mat maos;
    int threshold_value = 0;
    int type = 3;

    XnStatus nRetVal = XN_STATUS_OK;
    xn::EnumerationErrors errors;

    nRetVal = g_Context.Init();
    xn::Player xPlayer;
    nRetVal = g_Context.OpenFileRecording(argv[1], xPlayer); // read the oni file
    //xPlayer.SetRepeat(false);


    if (nRetVal == XN_STATUS_NO_NODE_PRESENT)
    {
        XnChar strError[1024];
        errors.ToString(strError, 1024);
        printf("%s\n", strError);
        return (nRetVal);
    }
    else if (nRetVal != XN_STATUS_OK)
    {
        printf("Open failed: %s\n", xnGetStatusString(nRetVal));
        return (nRetVal);
    }

    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_DepthGenerator);
    CHECK_RC(nRetVal,"No depth");

    xn::DepthMetaData g_depthMD ;
    g_DepthGenerator.GetMetaData(g_depthMD);

    XnUInt32 frames;
    xPlayer.GetNumFrames(g_DepthGenerator.GetName(), frames);

    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_USER, g_UserGenerator);
    if (nRetVal != XN_STATUS_OK)
    {
        nRetVal = g_UserGenerator.Create(g_Context);
        CHECK_RC(nRetVal, "Find user generator");
    }

    if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_SKELETON))
    {
        // If the UserGenerator cannot support skeleton tracking
        printf("Supplied user generator doesn't support skeleton\n");
        return 1;
    }

    //Register the Callbacks.When a new user enters the scene,start callibration.If pose is reqired,search for pose,find it and then start callibration
    XnCallbackHandle hUserCallbacks, hCalibrationStart, hCalibrationComplete, hPoseDetected,hUserExit,hUserReEnter;

    nRetVal = g_UserGenerator.RegisterUserCallbacks(User_NewUser, User_LostUser, NULL, hUserCallbacks);
    CHECK_RC(nRetVal, "Register to user callbacks");

     nRetVal = g_UserGenerator.RegisterToUserExit(UserExit,NULL,hUserExit);
     CHECK_RC(nRetVal, "Register to user Exit");

     nRetVal = g_UserGenerator.RegisterToUserExit(UserReEnter,NULL,hUserReEnter);
     CHECK_RC(nRetVal, "Register to user Reenter");

    nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationStart(UserCalibration_CalibrationStart, NULL, hCalibrationStart);
    CHECK_RC(nRetVal, "Register to calibration start");
    nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationComplete(UserCalibration_CalibrationComplete, NULL, hCalibrationComplete);
    CHECK_RC(nRetVal, "Register to calibration complete");

    if (g_UserGenerator.GetSkeletonCap().NeedPoseForCalibration())
    {
        //Check whether Pose is required for callibration

        g_bNeedPose = TRUE;
        if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_POSE_DETECTION))
        {
            // If Pose is required and not supported,end program
            printf("Pose required, but not supported\n");
            return 1;
        }

        //Else Register the PoseDetection Callback
        nRetVal = g_UserGenerator.GetPoseDetectionCap().RegisterToPoseDetected(UserPose_PoseDetected, NULL, hPoseDetected);
        CHECK_RC(nRetVal, "Register to Pose Detected");

        //Check for Pose defined by the g_strPose.Wait till pose is found.Once found,run the UserPose_PoseDetected callback
        //That callback starts callibration of the User.
        g_UserGenerator.GetSkeletonCap().GetCalibrationPose(g_strPose);
    }

    // What are of the user should be tracked.
   // XN_SKEL_PROFILE_UPPER
   // XN_SKEL_PROFILE_LOWER
    //XN_SKEL_PROFILE_ALL
    //XN_SKEL_PROFILE_HEAD_HANDS
    g_UserGenerator.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);

    nRetVal = g_Context.StartGeneratingAll(); //Start generating Data
    CHECK_RC(nRetVal, "StartGenerating");

    XnUserID aUsers[MAX_NUM_USERS];
    XnUInt16 nUsers;
    XnSkeletonJointTransformation torsoJoint;

    printf("Starting to run\n");
    if(g_bNeedPose)
    {
        printf("Assume calibration pose\n");
    }

    colorArr[0]  = cv::Mat(g_depthMD.YRes(),g_depthMD.XRes(),CV_8U);
    colorArr[1]  = cv::Mat(g_depthMD.YRes(),g_depthMD.XRes(),CV_8U);
    colorArr[2]  = cv::Mat(g_depthMD.YRes(),g_depthMD.XRes(),CV_8U);

    XnUInt32 epochTime = 0;
    bool endit = false;
    bool track = true;
    while (!endit)
    {

        if(xnOSWasKeyboardHit())
        {
            //Decide whether to track or not
            char c = xnOSReadCharFromInput();
            if(c == 27) endit = !endit;
            else if (c == 32) track = !track;
        }

        g_Context.WaitOneUpdateAll(g_UserGenerator);
        // wait for UserData to be updated
        nUsers=MAX_NUM_USERS;
        //Get all the users in the scene,and the number of users
        g_UserGenerator.GetUsers(aUsers, nUsers);
        int numTracked=0;
        int userToPrint=-1;

// --------------------------------------
//Method to convert raw Depth Image into a colourfull Depth Histogram
//---------------------------------------
    const XnDepthPixel* pDepth = g_depthMD.Data();
	xnOSMemSet(g_pDepthHist, 0, MAX_DEPTH*sizeof(float));

	unsigned int nNumberOfPoints = 0;
	for (XnUInt y = 0; y < g_depthMD.YRes(); ++y)
	{
		for (XnUInt x = 0; x < g_depthMD.XRes(); ++x, ++pDepth)
		{
			if (*pDepth != 0)
			{
				g_pDepthHist[*pDepth]++;
				nNumberOfPoints++;
			}
		}
	}
	for (int nIndex=1; nIndex<MAX_DEPTH; nIndex++)
	{
		g_pDepthHist[nIndex] += g_pDepthHist[nIndex-1];
	}
	if (nNumberOfPoints)
	{
		for (int nIndex=1; nIndex<MAX_DEPTH; nIndex++)
		{
		    //Histogram creation
			g_pDepthHist[nIndex] = (unsigned int)(256 * (1.0f - (g_pDepthHist[nIndex] / nNumberOfPoints)));
		}
	}

        // Converting the Histogram into an OpenCV image/Mat
        const XnDepthPixel* pDepthRow = g_depthMD.Data();
        for (XnUInt y = 0; y < g_depthMD.YRes(); ++y)
		{
			const XnDepthPixel* pDepth = pDepthRow;
			uchar* Bptr = colorArr[0].ptr<uchar>(y);
			uchar* Gptr = colorArr[1].ptr<uchar>(y);
			uchar* Rptr = colorArr[2].ptr<uchar>(y);
			for (XnUInt x = 0; x < g_depthMD.XRes(); ++x, ++pDepth)
			{
				 int nHistValue = g_pDepthHist[*pDepth];
				 Bptr[x] = 0;
				 Gptr[x] = 0;
				 Rptr[x] = nHistValue;

			}

			pDepthRow += g_depthMD.XRes();
			}
			cv::merge(colorArr,3,colorImg);

		if(track == true)
		{
		    for(XnUInt16 i=0; i<nUsers; i++)
        {
            if(g_UserGenerator.GetSkeletonCap().IsTracking(aUsers[i])==FALSE)
                continue;

            frames++;
            XnPoint3D com;
            //User Center of Mass
            g_UserGenerator.GetCoM(aUsers[i],com);
            //printf("%f  ,  %f  , %f\n ",com.X/10,com.Y/10,com.Z/10);
            //User head Joint
            g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_HEAD,torsoJoint);

            XnPoint3D aProjective;
            //Convert the positions from OpenNI coordinate system into Projective/OpenCV coordinate system
            g_DepthGenerator.ConvertRealWorldToProjective(1,&torsoJoint.position.position,&aProjective);
            int x1 = aProjective.X;
            int y1 = aProjective.Y;

            //printf("\nHead at %d , %d",x1,y1);

               g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_NECK,torsoJoint);

                g_DepthGenerator.ConvertRealWorldToProjective(1,&torsoJoint.position.position,&aProjective);
            int x2 = aProjective.X;
            int y2 = aProjective.Y;
            //printf("\nNeck at %d,%d",x2,y2);

             g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_RIGHT_COLLAR,torsoJoint);

                g_DepthGenerator.ConvertRealWorldToProjective(1,&torsoJoint.position.position,&aProjective);
            int x3 = aProjective.X;
            int y3 = aProjective.Y;
            //printf("\nL-Collar at %d,%d",x3,y3);

            g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_RIGHT_ELBOW,torsoJoint);

                g_DepthGenerator.ConvertRealWorldToProjective(1,&torsoJoint.position.position,&aProjective);
            int x4 = aProjective.X;
            int y4 = aProjective.Y;
            //printf("\nL-Elbow at %d,%d",x4,y4);

             g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_RIGHT_HAND,torsoJoint);

            g_DepthGenerator.ConvertRealWorldToProjective(1,&torsoJoint.position.position,&aProjective);
            int x5 = aProjective.X;
            int y5 = aProjective.Y;
            int z5 = aProjective.Z;
           // printf("\nL-Hand at %d,%d\n",x5,y5);

            g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_TORSO,torsoJoint);

            g_DepthGenerator.ConvertRealWorldToProjective(1,&torsoJoint.position.position,&aProjective);
            int x6 = aProjective.X;
            int y6 = aProjective.Y;
            int z6 = aProjective.Z;
           // printf("\nL-Hand at %d,%d\n",x5,y5);

           g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_LEFT_COLLAR,torsoJoint);

            g_DepthGenerator.ConvertRealWorldToProjective(1,&torsoJoint.position.position,&aProjective);
            int x7 = aProjective.X;
            int y7 = aProjective.Y;

             g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_LEFT_ELBOW,torsoJoint);

            g_DepthGenerator.ConvertRealWorldToProjective(1,&torsoJoint.position.position,&aProjective);
            int x8 = aProjective.X;
            int y8 = aProjective.Y;

            g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_LEFT_HAND,torsoJoint);

            g_DepthGenerator.ConvertRealWorldToProjective(1,&torsoJoint.position.position,&aProjective);
            int x9 = aProjective.X;
            int y9 = aProjective.Y;
            int z9 = aProjective.Z;

            //Frame(int rhX, int rhY, int rhZ, int hc1, int lhX, int lhY, int lhZ, int hc2, int tX, int tY, int tZ, int hX, int hY)
            Frame cFrame(x5, y5, z5, 1, x9, y9, z9, 1, x6, y6, z6, x1, y1);
            cout << cFrame.rightVectorX << "\t" << cFrame.rightVectorY << "\t" << cFrame.rightVectorZ << "\t" << cFrame.handConfigurationRight << "\t" << cFrame.leftVectorX << "\t" << cFrame.leftVectorY << "\t" << cFrame.leftVectorZ << "\t" << cFrame.handConfigurationLeft << endl;
            cFrame.WriteToFile(dataset);
                
            
            ///////Cria uma janela em volta das mãos
            float roiSize = 75;
            cv::Rect roi(x5 - (roiSize/2), y5 - (roiSize/2), roiSize, roiSize);
            cv::Rect roiImg(0, 0, colorImg.cols, colorImg.rows);

            //maos = cv::imread("default.jpg");
            //Checa para ver se esta "dentro" da imagem
            if( (roi.area() > 0) && ((roiImg & roi).area() == roi.area()) )
                maos = colorImg(roi);    


            cv::cvtColor(maos,maos,CV_RGB2GRAY);
            //namedWindow("Threshold Demo", CV_WINDOW_AUTOSIZE);
            //createTrackbar("Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted","Threshold Demo", &type, 4);  
            //createTrackbar("Value", "Threshold Demo", &threshold_value, 255);
            //threshold(maos, maos, threshold_value, 255, type);
            //adaptiveThreshold(maos, maos, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2);
            threshold(maos, maos, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            ///////


                
                //making a stick figure

                cv::line(colorImg,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(255,255,255),2);

                //cv::line(colorImg,cv::Point(x7,y7),cv::Point(x2,y2),cv::Scalar(255,255,255),2);
                //cv::line(colorImg,cv::Point(x7,y7),cv::Point(x8,y8),cv::Scalar(255,255,255),2);
                //cv::line(colorImg,cv::Point(x9,y9),cv::Point(x8,y8),cv::Scalar(255,255,255),2);

                cv::line(colorImg,cv::Point(x3,y3),cv::Point(x2,y2),cv::Scalar(255,255,255),2);
                cv::line(colorImg,cv::Point(x3,y3),cv::Point(x4,y4),cv::Scalar(255,255,255),2);
                cv::line(colorImg,cv::Point(x5,y5),cv::Point(x4,y4),cv::Scalar(255,255,255),2);

                cv::line(colorImg,cv::Point(x6,y6),cv::Point(x2,y2),cv::Scalar(255,255,255),2);
                //cv::line(colorImg,cv::Point(x6,y6),cv::Point(x7,y7),cv::Scalar(255,255,255),2);
                cv::line(colorImg,cv::Point(x6,y6),cv::Point(x3,y3),cv::Scalar(255,255,255),2);


                cv::line(colorImg,cv::Point(x6,y6),cv::Point((int)(x6+cFrame.rightVectorX),(int)(y6+cFrame.rightVectorY)),cv::Scalar(0,255,0),2);
                cv::line(colorImg,cv::Point(x6,y6),cv::Point((int)(x6+cFrame.leftVectorX),(int)(y6+cFrame.leftVectorY)),cv::Scalar(0,255,0),2);


        }
		}

			cv::imshow("HIST",colorImg);
            if(!maos.empty())
                cv::imshow("HAND", maos);
			char esc = cv::waitKey(33);
			if (esc == 27) break;
			else if (esc == 32)
			{
			    //cv::imwrite("DepthMap.jpg",colorImg);
			    track = !track;
			}

    }
    cout << "Frames: " << frames << endl;
    dataset.close(); 
    g_scriptNode.Release();
    g_DepthGenerator.Release();
    g_UserGenerator.Release();
    g_Context.Release();

}



























/*
#define MAX_NUM_USERS 15


int main(int argc, char* argv[]){
    if(argc < 2){
        cerr << "Uso: " << argv[0] << " <oni file>" << endl;
        return -1;
    }

    XnStatus nRetVal = g_Context.Init();
    if(nRetVal != XN_STATUS_OK){
        cerr << "Erro no g_Context.Init()" << endl;
        return -1;
    }



    nRetVal = g_Context.OpenFileRecording(argv[1]); // read the oni file
    if (nRetVal == XN_STATUS_NO_NODE_PRESENT){
        cerr << "No node present" << endl;
        return -1;
    }
    else if (nRetVal != XN_STATUS_OK){
        printf("Open failed: %s\n", xnGetStatusString(nRetVal));
        return (nRetVal);
    }



    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_DepthGenerator);
    if(nRetVal != XN_STATUS_OK){
        cerr << "Impossível iniciar, pois não há um nó de profundidade presente na gravação" << endl;
        return -1;
    }

    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_USER, g_UserGenerator);
    if(nRetVal != XN_STATUS_OK){
        nRetVal = g_UserGenerator.Create(g_Context);
    }


    XnUserID aUsers[MAX_NUM_USERS];
    XnUInt16 nUsers;
    XnSkeletonJointPosition handPosition;

    nUsers=MAX_NUM_USERS;
    g_UserGenerator.GetUsers(aUsers, nUsers);
    for(XnUInt16 i=0; i<nUsers; i++)
    {
        if(g_UserGenerator.GetSkeletonCap().IsTracking(aUsers[i])==FALSE)
            continue;

        //g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_LEFT_HAND,torsoJoint);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_LEFT_HAND, handPosition);
    }
}*/