#include "stdafx.h"
#include "windows.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/warpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;


#define LOGLN(msg) std::cout << msg << std::endl

#pragma region Algorithim Parasmeters
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
string features_type = "orb";
float match_conf = 0.3f;
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
int range_width = -1;
#pragma endregion


void stitchCamera(std::string cameraIDs, int frameWidth, int frameHeight)
{

#pragma region Intermediate Variables
	size_t num_cameras = cameraIDs.size();

	bool find_features = true;

	VideoCapture cap;
	vector<VideoCapture> caps;

	float warped_image_scale;
	double work_scale, seam_scale, compose_scale, seam_work_aspect, compose_work_aspect;
	bool is_work_scale_set, is_seam_scale_set, is_compose_scale_set;

	Mat img, full_img, img_warped, img_warped_s, mask, mask_warped, dilated_mask, seam_mask;

	Ptr<RotationWarper> warper;
	Ptr<WarperCreator> warper_creator;
	Ptr<ExposureCompensator> compensator;

	vector<int> indices;
	vector<Point> corners(num_cameras);
	vector<Size> sizes(num_cameras);
	vector<Size> full_img_sizes(num_cameras);
	vector<Mat> camera_images(num_cameras);
	vector<UMat> masks_warped(num_cameras);
	vector<CameraParams> cameras;
#pragma endregion

	// Initialize all cameras
	for (size_t i = 0; i < num_cameras; ++i)
	{
		cap = VideoCapture(int(cameraIDs[i]) - 48);
		cap.set(CAP_PROP_FRAME_WIDTH, frameWidth);
		cap.set(CAP_PROP_FRAME_HEIGHT, frameHeight);
		caps.push_back(cap);
	}

	while (true)
	{
		long t0 = GetTickCount();

		// Check if all cameras are opened
		for (size_t i = 0; i < num_cameras; ++i)
		{
			if (!caps[i].isOpened())
			{
				LOGLN("Open camera failed");
				goto error;
			}
		}

#pragma region Feature extraction and transformation parameter calculation
		// Only executed on the first frame and when Enter is pressed
		if (find_features)
		{
			work_scale = 1;
			seam_scale = 1;
			compose_scale = 1;
			is_work_scale_set = false;
			is_seam_scale_set = false;
			is_compose_scale_set = false;

			LOGLN("Finding features...");

			Ptr<Feature2D> finder;
			if (features_type == "orb")
			{
				finder = ORB::create();
			}
			else if (features_type == "akaze")
			{
				finder = AKAZE::create();
			}
			else
			{
				LOGLN("Unknown 2D features type: '" << features_type);
				goto error;
			}

			vector<ImageFeatures> features(num_cameras);
			vector<Mat> images(num_cameras);

			seam_work_aspect = 1;

			for (int i = 0; i < num_cameras; ++i)
			{
				caps[i].read(full_img);
				full_img_sizes[i] = full_img.size();

				if (full_img.empty())
				{
					LOGLN("Read frame failed");
					goto error;
				}
				if (work_megapix < 0)
				{
					img = full_img;
					work_scale = 1;
					is_work_scale_set = true;
				}
				else
				{
					if (!is_work_scale_set)
					{
						work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
						is_work_scale_set = true;
					}
					resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
				}
				if (!is_seam_scale_set)
				{
					seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
					seam_work_aspect = seam_scale / work_scale;
					is_seam_scale_set = true;
				}

				computeImageFeatures(finder, img, features[i]);
				features[i].img_idx = i;
				LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());

				resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
				images[i] = img.clone();
			}
			full_img.release();
			img.release();

			LOGLN("Pairwise matching");

			vector<MatchesInfo> pairwise_matches;
			Ptr<FeaturesMatcher> matcher;
			if (matcher_type == "affine")
				matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
			else if (range_width == -1)
				matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
			else
				matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);

			(*matcher)(features, pairwise_matches);
			matcher->collectGarbage();

			// Leave only images we are sure are from the same panorama
			indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
			vector<Mat> img_subset;
			vector<Size> full_img_sizes_subset;
			for (size_t i = 0; i < indices.size(); ++i)
			{
				img_subset.push_back(images[indices[i]]);
				full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
			}

			images = img_subset;
			full_img_sizes = full_img_sizes_subset;

			// Check if we still have enough images
			if (indices.size() < num_cameras)
			{
				LOGLN("Insufficient stitchable images");
				continue;
			}

			Ptr<Estimator> estimator;
			if (estimator_type == "affine")
				estimator = makePtr<AffineBasedEstimator>();
			else
				estimator = makePtr<HomographyBasedEstimator>();

			if (!(*estimator)(features, pairwise_matches, cameras))
			{
				LOGLN("Homography estimation failed");
				continue;
			}

			for (size_t i = 0; i < cameras.size(); ++i)
			{
				Mat R;
				cameras[i].R.convertTo(R, CV_32F);
				cameras[i].R = R;
				LOGLN("Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
			}

			Ptr<detail::BundleAdjusterBase> adjuster;
			if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
			else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
			else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
			else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
			else
			{
				LOGLN("Unknown bundle adjustment cost function: '" << ba_cost_func);
				continue;
			}
			adjuster->setConfThresh(conf_thresh);
			Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
			if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
			if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
			if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
			if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
			if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
			adjuster->setRefinementMask(refine_mask);
			if (!(*adjuster)(features, pairwise_matches, cameras))
			{
				LOGLN("Camera parameters adjusting failed");
				continue;
			}

			// Find median focal length

			vector<double> focals;
			for (size_t i = 0; i < cameras.size(); ++i)
			{
				LOGLN("Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
				focals.push_back(cameras[i].focal);
			}

			sort(focals.begin(), focals.end());

			if (focals.size() % 2 == 1)
				warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
			else
				warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

			if (do_wave_correct)
			{
				vector<Mat> rmats;
				for (size_t i = 0; i < cameras.size(); ++i)
					rmats.push_back(cameras[i].R.clone());
				waveCorrect(rmats, wave_correct);
				for (size_t i = 0; i < cameras.size(); ++i)
					cameras[i].R = rmats[i];
			}

			LOGLN("Warping images (auxiliary)... ");

			vector<UMat> images_warped(num_cameras);

			vector<UMat> masks(num_cameras);

			// Prepare images masks
			for (int i = 0; i < num_cameras; ++i)
			{
				masks[i].create(images[i].size(), CV_8U);
				masks[i].setTo(Scalar::all(255));
			}

			// Warp images and their masks


#ifdef HAVE_OPENCV_CUDAWARPING
			if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
			{
				if (warp_type == "plane")
					warper_creator = makePtr<cv::PlaneWarperGpu>();
				else if (warp_type == "cylindrical")
					warper_creator = makePtr<cv::CylindricalWarperGpu>();
				else if (warp_type == "spherical")
					warper_creator = makePtr<cv::SphericalWarperGpu>();
			}
			else
#endif
			{
				if (warp_type == "plane")
					warper_creator = makePtr<cv::PlaneWarper>();
				else if (warp_type == "affine")
					warper_creator = makePtr<cv::AffineWarper>();
				else if (warp_type == "cylindrical")
					warper_creator = makePtr<cv::CylindricalWarper>();
				else if (warp_type == "spherical")
					warper_creator = makePtr<cv::SphericalWarper>();
				else if (warp_type == "fisheye")
					warper_creator = makePtr<cv::FisheyeWarper>();
				else if (warp_type == "stereographic")
					warper_creator = makePtr<cv::StereographicWarper>();
				else if (warp_type == "compressedPlaneA2B1")
					warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
				else if (warp_type == "compressedPlaneA1.5B1")
					warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
				else if (warp_type == "compressedPlanePortraitA2B1")
					warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
				else if (warp_type == "compressedPlanePortraitA1.5B1")
					warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
				else if (warp_type == "paniniA2B1")
					warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
				else if (warp_type == "paniniA1.5B1")
					warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
				else if (warp_type == "paniniPortraitA2B1")
					warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
				else if (warp_type == "paniniPortraitA1.5B1")
					warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
				else if (warp_type == "mercator")
					warper_creator = makePtr<cv::MercatorWarper>();
				else if (warp_type == "transverseMercator")
					warper_creator = makePtr<cv::TransverseMercatorWarper>();
			}

			if (!warper_creator)
			{
				LOGLN("Can't create the following warper '" << warp_type);
				continue;
			}

			warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

			for (int i = 0; i < num_cameras; ++i)
			{
				Mat_<float> K;
				cameras[i].K().convertTo(K, CV_32F);
				float swa = (float)seam_work_aspect;
				K(0, 0) *= swa; K(0, 2) *= swa;
				K(1, 1) *= swa; K(1, 2) *= swa;

				corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
				sizes[i] = images_warped[i].size();

				warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
			}

			vector<UMat> images_warped_f(num_cameras);
			for (int i = 0; i < num_cameras; ++i)
				images_warped[i].convertTo(images_warped_f[i], CV_32F);

			LOGLN("Compensating exposure...");

			compensator = ExposureCompensator::createDefault(expos_comp_type);
			if (dynamic_cast<GainCompensator*>(compensator.get()))
			{
				GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
				gcompensator->setNrFeeds(expos_comp_nr_feeds);
			}

			if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
			{
				ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
				ccompensator->setNrFeeds(expos_comp_nr_feeds);
			}

			if (dynamic_cast<BlocksCompensator*>(compensator.get()))
			{
				BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
				bcompensator->setNrFeeds(expos_comp_nr_feeds);
				bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
				bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
			}

			compensator->feed(corners, images_warped, masks_warped);

			LOGLN("Finding seams...");

			Ptr<SeamFinder> seam_finder;
			if (seam_find_type == "no")
				seam_finder = makePtr<detail::NoSeamFinder>();
			else if (seam_find_type == "voronoi")
				seam_finder = makePtr<detail::VoronoiSeamFinder>();
			else if (seam_find_type == "gc_color")
			{
#ifdef HAVE_OPENCV_CUDALEGACY
				if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
					seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
				else
#endif
					seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
			}
			else if (seam_find_type == "gc_colorgrad")
			{
#ifdef HAVE_OPENCV_CUDALEGACY
				if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
					seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
				else
#endif
					seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
			}
			else if (seam_find_type == "dp_color")
				seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
			else if (seam_find_type == "dp_colorgrad")
				seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
			if (!seam_finder)
			{
				LOGLN("Can't create the following seam finder '" << seam_find_type);
				continue;
			}

			seam_finder->find(images_warped_f, corners, masks_warped);

			// Release unused memory
			images.clear();
			images_warped.clear();
			images_warped_f.clear();
			masks.clear();

			LOGLN("Compositing...");

			compose_work_aspect = 1;
			find_features = false;
		}
#pragma endregion

#pragma region Image blending and display
		Ptr<Blender> blender;
		for (int img_idx = 0; img_idx < num_cameras; ++img_idx)
		{
			LOGLN("Compositing image #" << indices[img_idx] + 1);

			// Read image and resize it if necessary
			caps[img_idx].read(full_img);
			camera_images[img_idx] = full_img;

			if (!is_compose_scale_set)
			{
				if (compose_megapix > 0)
					compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
				is_compose_scale_set = true;

				// Compute relative scales
				compose_work_aspect = compose_scale / work_scale;

				// Update warped image scale
				warped_image_scale *= static_cast<float>(compose_work_aspect);
				warper = warper_creator->create(warped_image_scale);

				// Update corners and sizes
				for (int i = 0; i < num_cameras; ++i)
				{
					// Update intrinsics
					cameras[i].focal *= compose_work_aspect;
					cameras[i].ppx *= compose_work_aspect;
					cameras[i].ppy *= compose_work_aspect;

					// Update corner and size
					Size sz = full_img_sizes[i];
					if (std::abs(compose_scale - 1) > 1e-1)
					{
						sz.width = cvRound(full_img_sizes[i].width * compose_scale);
						sz.height = cvRound(full_img_sizes[i].height * compose_scale);
					}

					Mat K;
					cameras[i].K().convertTo(K, CV_32F);
					Rect roi = warper->warpRoi(sz, K, cameras[i].R);
					corners[i] = roi.tl();
					sizes[i] = roi.size();
				}
			}
			if (abs(compose_scale - 1) > 1e-1)
				resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
			else
				img = full_img;
			full_img.release();
			Size img_size = img.size();

			Mat K;
			cameras[img_idx].K().convertTo(K, CV_32F);

			// Warp the current image
			warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

			// Warp the current image mask
			mask.create(img_size, CV_8U);
			mask.setTo(Scalar::all(255));
			warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

			// Compensate exposure
			compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

			img_warped.convertTo(img_warped_s, CV_16S);
			img_warped.release();
			img.release();
			mask.release();

			dilate(masks_warped[img_idx], dilated_mask, Mat());
			resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
			mask_warped = seam_mask & mask_warped;

			if (!blender)
			{
				blender = Blender::createDefault(blend_type, try_cuda);
				Size dst_sz = resultRoi(corners, sizes).size();
				float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
				if (blend_width < 1.f)
					blender = Blender::createDefault(Blender::NO, try_cuda);
				else if (blend_type == Blender::MULTI_BAND)
				{
					MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
					mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
					LOGLN("Multi-band blender, number of bands: " << mb->numBands());
				}
				else if (blend_type == Blender::FEATHER)
				{
					FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
					fb->setSharpness(1.f / blend_width);
					LOGLN("Feather blender, sharpness: " << fb->sharpness());
				}
				blender->prepare(corners, sizes);
			}

			// Blend the current image
			blender->feed(img_warped_s, mask_warped, corners[img_idx]);
		}

		Mat result, result_mask;
		blender->blend(result, result_mask);
		result.convertTo(result, CV_8U);

		long t1 = GetTickCount();
		putText(result, "FPS: " + to_string(int(1000 / (t1 - t0))), Point(100, 100), FONT_HERSHEY_COMPLEX, 1, (0, 255, 0));
		imshow("Panaromic", result);

		int key_num = waitKey(1);
		if (key_num == 13)
		{
			find_features = true;
		}
		else if (key_num == 27)
		{
			break;
		}
#pragma endregion
	}
error:
	{
		LOGLN("Process finished");
	}
}