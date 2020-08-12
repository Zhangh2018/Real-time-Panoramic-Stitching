#pragma once
#include "targetver.h"
#include <iostream>
#include <stdio.h>
#include <tchar.h>


/** @brief Stitch the images into a panoramic image.

The function reads all the images to be stitched and tries to stitch as a panoramic image to save.

@param imgNames File names of all images to be stitched separated by commas.
*/
void stitchImage(std::string imgNames);


/** @brief Real-time panoramic stitching of multiple cameras.

The function opens a number of specified cameras and tries to stitch the frames as a panoramic video stream for real-time display.

@param cameraIDs IDs of all specified cameras.
@param frameWidth Width of camera frames.
@param frameHeight Height of camera frames.
*/
void stitchCamera(std::string cameraIDs, int frameWidth = 1280, int frameHeight = 720);