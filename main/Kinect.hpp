#ifndef KINECT_H
#define KINECT_H

//-----------------------------------------------------------------------
//  Includes
//-----------------------------------------------------------------------
#include <XnCppWrapper.h>
#include "kmeans.hpp"

//-----------------------------------------------------------------------
//  Namespaces
//-----------------------------------------------------------------------
using namespace std;
using namespace xn;

//-----------------------------------------------------------------------
//  Defines
//-----------------------------------------------------------------------
#define SAMPLE_XML_PATH "../../../../Data/SamplesConfig.xml"
#define SAMPLE_XML_PATH_LOCAL "SamplesConfig.xml"
#define MAX_NUM_USERS 15
#define CHECK_RC(nRetVal, what)                     \
    if (nRetVal != XN_STATUS_OK)                    \
{                                   \
    printf("%s failed: %s\n", what, xnGetStatusString(nRetVal));    \
    return nRetVal;                         \
}

//-----------------------------------------------------------------------
//  Code
//-----------------------------------------------------------------------

XnBool fileExists(const char *fn)
{
    XnBool exists;
    xnOSDoesFileExist(fn, &exists);
    return exists;
}

Context g_Context;
ScriptNode g_ScriptNode;
UserGenerator g_UserGenerator;
DepthGenerator g_DepthGenerator;

XnBool g_bNeedPose;
XnChar g_strPose[20];

VideoCapture cap;

// Callback: New user was detected
void XN_CALLBACK_TYPE User_NewUser(xn::UserGenerator& /*generator*/, XnUserID nId, void* /*pCookie*/){
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d New User %d\n", epochTime, nId);
    // New user found
    if (g_bNeedPose)
        g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
    else
        g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
}

// Callback: An existing user was lost
void XN_CALLBACK_TYPE User_LostUser(xn::UserGenerator& /*generator*/, XnUserID nId, void* /*pCookie*/){
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Lost user %d\n", epochTime, nId);
}

// Callback: Detected a pose
void XN_CALLBACK_TYPE UserPose_PoseDetected(xn::PoseDetectionCapability& /*capability*/, const XnChar* strPose, XnUserID nId, void* /*pCookie*/){
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Pose %s detected for user %d\n", epochTime, strPose, nId);
    g_UserGenerator.GetPoseDetectionCap().StopPoseDetection(nId);
    g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
}

// Callback: Started calibration
void XN_CALLBACK_TYPE UserCalibration_CalibrationStart(xn::SkeletonCapability& /*capability*/, XnUserID nId, void* /*pCookie*/){
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Calibration started for user %d\n", epochTime, nId);
}

void XN_CALLBACK_TYPE UserCalibration_CalibrationComplete(xn::SkeletonCapability& /*capability*/, XnUserID nId, XnCalibrationStatus eStatus, void* /*pCookie*/){
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    if (eStatus == XN_CALIBRATION_STATUS_OK){
        // Calibration succeeded
        printf("%d Calibration complete, start tracking user %d\n", epochTime, nId);        
        g_UserGenerator.GetSkeletonCap().StartTracking(nId);
    }
    else{
        // Calibration failed
        printf("%d Calibration failed for user %d\n", epochTime, nId);
        if(eStatus==XN_CALIBRATION_STATUS_MANUAL_ABORT){
            printf("Manual abort occured, stop attempting to calibrate!");
            return;
        }
        if (g_bNeedPose)
            g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
        else
            g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
    }
}


XnStatus Inicializar(){
    XnStatus nRetVal = XN_STATUS_OK;
    xn::EnumerationErrors errors;

    const char *fn = NULL;
    if    (fileExists(SAMPLE_XML_PATH)) fn = SAMPLE_XML_PATH;
    else if (fileExists(SAMPLE_XML_PATH_LOCAL)) fn = SAMPLE_XML_PATH_LOCAL;
    else {
        printf("Could not find '%s' nor '%s'. Aborting.\n" , SAMPLE_XML_PATH, SAMPLE_XML_PATH_LOCAL);
        return XN_STATUS_ERROR;
    }
    printf("Reading config from: '%s'\n", fn);

    nRetVal = g_Context.InitFromXmlFile(fn, g_ScriptNode, &errors);
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

    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_USER, g_UserGenerator);
    if (nRetVal != XN_STATUS_OK)
    {
        nRetVal = g_UserGenerator.Create(g_Context);
        CHECK_RC(nRetVal, "Find user generator");
    }

    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_DepthGenerator);
    if (nRetVal != XN_STATUS_OK)
    {
        nRetVal = g_DepthGenerator.Create(g_Context);
        CHECK_RC(nRetVal, "Find depth generator");
    }

    XnCallbackHandle hUserCallbacks, hCalibrationStart, hCalibrationComplete, hPoseDetected;
    if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_SKELETON))
    {
        printf("Supplied user generator doesn't support skeleton\n");
        return XN_STATUS_ERROR;
    }
    nRetVal = g_UserGenerator.RegisterUserCallbacks(User_NewUser, User_LostUser, NULL, hUserCallbacks);
    CHECK_RC(nRetVal, "Register to user callbacks");
    nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationStart(UserCalibration_CalibrationStart, NULL, hCalibrationStart);
    CHECK_RC(nRetVal, "Register to calibration start");
    nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationComplete(UserCalibration_CalibrationComplete, NULL, hCalibrationComplete);
    CHECK_RC(nRetVal, "Register to calibration complete");

    if (g_UserGenerator.GetSkeletonCap().NeedPoseForCalibration())
    {
        g_bNeedPose = TRUE;
        if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_POSE_DETECTION))
        {
            printf("Pose required, but not supported\n");
            return XN_STATUS_ERROR;
        }
        nRetVal = g_UserGenerator.GetPoseDetectionCap().RegisterToPoseDetected(UserPose_PoseDetected, NULL, hPoseDetected);
        CHECK_RC(nRetVal, "Register to Pose Detected");
        g_UserGenerator.GetSkeletonCap().GetCalibrationPose(g_strPose);
    }

    g_UserGenerator.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);

    nRetVal = g_Context.StartGeneratingAll();
    CHECK_RC(nRetVal, "StartGenerating");

    return nRetVal;
}







bool createKinect(){
    if(Inicializar() != XN_STATUS_OK){
        cerr << "Erro ao Inicializar o Kinect." << endl;
        return false;
    }
    return true;
}

bool createOpenCV(){
    VideoCapture capture(CV_CAP_OPENNI);
    capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ);
    if(!capture.isOpened()){
        cerr << "Erro ao Inicializar OpenCV com OpenNI." << endl;
        return false;
    }

    cap = capture;
    return true;
}


/*
void drawSkeleton(){
    g_UserGenerator.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);
    
    XnUserID aUsers[MAX_NUM_USERS];
    XnUInt16 nUsers;


    g_UserGenerator.GetUsers(users, nUsers);

    for(XnUInt16 i = 0; i < MAX_NUM_USERS; i++){
        if(!g.UserGenerator.GetSkeletonCap().IsTracking(users[i]))
            continue;

        g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i], XN_SKEL_LEFT_HIP, )
    }

}*/

#endif //KINECT_H