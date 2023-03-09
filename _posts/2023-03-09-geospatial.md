---
title: "Geospatial survey for EO, UAV tasks in AI/Computer vision"
excerpt: "I am machine learning practitioner I am mailny focused in Computer vision area. For the past few months I have been interested in application of computer vision in geospatial domain. In this blog I want to showcase what I found, sources and my opinions for myself and people that also want to follow my path."
data: 2023-01-14
languages: [python]
tags: [Geospatial, Earth observation (EO), Unmanned aerial vehicle (UAV), Satellite images, Artificial intelligence (AI)]
toc: true
---

Before we dive into the main topic of this post, I wanted to briefly introduce you to [ControlNet :books:](https://arxiv.org/abs/2302.05543), a cool model that uses [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) to provide more control in image generation. With this model, you can input scribes or canny edges and ControlNet will produce images that reflect your input. 

In Fig. 1, you can see some examples of images I created using the [Hugging Face app :computer:](https://huggingface.co/spaces/hysts/ControlNet) with ControlNet. Please note that I used the lowest inference time to generate these images, but with better model tweaks and adjustments, it's possible to create even more realistic pictures that are on par with Stable Diffusion images. Now, let's move on to the main topic of this post.

<p align = "center">
<img src="/images/posts/geospatial_1_intro.png" width="600">
</p>
<p align = "center">
Figure 1 showcases two digital images created using ControlNet. The middle image was generated based on canny edges of the left image without any prompt, while the right image was created using pose estimation of the left image with the prompt "Data Scientist on the beach."
</p>

## Introduction :wave:

If you are a data scientist, machine learning engineer, or data engineer and want to start dealing with climate change issues or work with satellite and drone images, you're in the right place :facepunch:. In this article, we will mainly focus on machine learning and computer vision techniques in geospatial data from [satellite imagery](https://en.wikipedia.org/wiki/Satellite_imagery), [Earth observation (EO)](https://en.wikipedia.org/wiki/Earth_observation_satellite) and [unmanned aerial vehicles (UAV)](https://en.wikipedia.org/wiki/Unmanned_aerial_vehicle).

If you are looking for information on geo-informatics and GIS then I will only lightly mention this topic, as I came across several articles and interactive books that covered this topic.

<p align = "center">
<img src="/images/posts/geospatial_1_detection.png" width="500">
</p>
<p align = "center">
Figure 2 Building detection results taken from this <a href="https://github.com/imenebak/OpenCv-Building-detection-from-Satellite-images">Github project</a>.
</p>

## GIS/Geography and Spatial Data Science

If you're interested in using spatial data to gain insights into patterns and processes, you may be interested in GIS and Geography, two related fields that focus on spatial data analysis.

[GIS (Geographic Information System)](https://education.nationalgeographic.org/resource/geographic-information-system-gis/) is a system used to capture, store, manipulate, analyze, and present spatial or geographic data, while Geography is a broader field that studies the physical and human aspects of the earth's surface, including their spatial relationships.

[Spatial Data Science](https://gisgeography.com/spatial-data-science/) is an interdisciplinary field that combines GIS, computer science, statistics, and machine learning to analyze large and complex spatial datasets. It focuses on developing and applying computational and statistical methods to gain insights into spatial patterns and processes and to inform decision-making.

If you want to learn more about Geography and Spatial Data Science, check out this [Jupyter Book :book:](https://geographicdata.science/book/intro.html) that showcases their applications and the tools/algorithms used, mainly in Python. The book provides many examples and codes, but some pages can be lengthy, making it difficult to read at times, yet still worth reading.

It's also useful to see how spatial data can be applied in practice. Check how Burger King used geospatial data to become the leader in fast food in Mexico a few years ago: 

<p align = "center">
<video src="https://www.youtube.com/watch?v=gwmd9HK8t4E" controls="controls" style="max-width: 1024px;">
</video>
</p>
<p align = "center">
Figure 3 Burger King traffic video.
</p>

You can also for starters, try working on projects such as Geographic Change Detection, Modeling Bike Ride Requests, Transport Demand Modeling, or Spatial Inequality Analysis.

If you want to learn more about modern geospatial approaches, I recommend reading [How to Learn Geospatial Data Science in 2023](https://towardsdatascience.com/how-to-learn-geospatial-data-science-in-2023-441d8386284e), which provides insights and tips on how to get started in this field. Additionally, this [source](https://gisgeography.com) provides a wealth of content for aspiring GIS professionals.

In my opinion, learning Python Geospatial data science is likely to be more in demand in the future, as Python is a popular language for AI and there is a big community with a lot of new tools. However, GIS and R are still commonly used today. This is just my intuition and opinion, but if I were to pursue this field, I would focus on Python Geospatial data science :information_desk_person:.


## Main dish :meat_on_bone: so Satellite Imagery

As I mentioned in the beginning, I will be focusing on satellite (and drone) images as it is a more applicable path for my skills in computer vision. I assume that you are already familiar with the field of computer vision and know what these tasks look like.

Satellite and drone images differ from other types of data that computer vision programmers typically work with because they are often [nadir](https://en.wikipedia.org/wiki/Nadir) format photos taken from the sky. From these photos, we can try to detect spatial features and primarily do some object detection or segmentation for further analysis.

Object detection in UAV images can be challenging for a few reasons. First, let's discuss satellite images. Most satellite images are already calibrated, meaning that they are nadir and stitched together to create a map covering vast land cover. The first problem we will encounter is concerning the resolution of these images, which is called Ground Sample Distance (GSD).

GSD is the size of one pixel in an image, measured in meters on the ground. GSD is determined by the sensor resolution and altitude of the satellite or drone. The smaller the GSD, the higher the resolution of the image.

However, detecting small objects can be difficult if we have high GSD. Conversely, if we have low GSD, we will need to zoom in more, but the image size will also be much larger, and we often don't work with images 2048x2048 but more like 512x512. Thus, sometimes we can miss some objects, or in an AI approach, we can analyze a region where not all important features are included. Based on this, we need to adjust resizing and zooming in.

---

An approach that I liked for dealing with small objects in object detection is not to detect but to estimate how many objects are in some grid box of the image model. It will give less information, but it will be more accurate for small objects or small resolution images. Here is the [medium paper :notebook:](https://medium.com/ecovisioneth/a-deep-learning-model-can-see-far-better-than-you-f689779eadf) presenting such solution for detecting trees. If you still would want to detect and not estimate small objects maybe this solution [SAHI :computer:](https://github.com/obss/sahi) may help you achieve it in inference.

The second major problem is clouds and cloud shades :cloud:. Images from satellites will often contain clouds that are covering the land, which will just not allow you to detect objects from this images or there are cloud shadows, making predictions difficult. There are some algorithms attempting to deal with this, and research is ongoing in this field. I advise you to look into this [paper dataset :pouch:](https://www.nature.com/articles/s41597-022-01878-2) about cloud satellite data.

<p align = "center">
<img src="/images/posts/geospatial_1_gsd.webp" width="500">
</p>
<p align = "center">
Figure 4 Comparison of different spatial resolutions.
</p>

In the case of segmentation, we have all the mentioned problems (clouds, resolution), and additionally, we can have problems with inconsistent predictions. For example, with line segmentation/detection or building segmentation, the masks will often not have consistent lines or structures. However, with post-processing, you can apply some treatment to it.

Now, let's talk about the temporal aspect. You need to understand that depending on the satellites, images taken of the globe are not all taken at the same time. Thus, some time shifts can happen, and it is even more problematic if you want to combine data from other satellites. They will not align in time, which can be very problematic when one satellite image shows a region without clouds and another shows the same region with clouds, or when there are significant changes in the spatial layout between the images.

### Optical Satellite Aspects

Data obtained from the satellites is not as simple as those obtained from our iphone or cameras and firstly I will tell about optical spectrum. RGB spectrum that we see is not only thing that can be seen however our eyes only capture this so why apparats need to capture more?

Sometimes we would like to see or analyse data from a different spectrum, for example, often in films soldiers use infrared goggles to see in the dark. Capturing data from a wider range of the electromagnetic spectrum can give us more information about some biological processes that we can't see with just that small band spectrum RGB.

<p align = "center">
<img src="/images/posts/geospatial_1_bands.png" width="600">
</p>
<p align = "center">
Figure 5 Spectral bands that satellite data often contains.
</p>

Let's look at the satelite bands from [sentinel 2](https://gisgeography.com/sentinel-2-bands-combinations/). Sentinel-2 is a European Union Earth Observation mission that consists of a constellation of two identical satellites, Sentinel-2A and Sentinel-2B. From the article you can what bands are gathered from satellite and what they can show and what combination of them can also show us.

Sentinel data are also preprocessed for the user but the version before preprocesing can also be obtained. It is good to know how data is pre-processed because various satellite data providers may include such things or may not and it good to know about them so we direct readers to the example about it for [sentinel 2](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/processing-levels/level-2).

There is also satellite SAR images that refer to images acquired by Synthetic Aperture Radar (SAR) systems onboard Earth observation satellites. SAR is a remote sensing technology that uses radar to generate high-resolution images of the Earth's surface, regardless of weather or lighting conditions.

<p align = "center">
<img src="/images/posts/geospatial_1_sar.png" width="500">
</p>
<p align = "center">
Figure 6 Satellite SAR image.
</p>

SAR can generate images with high spatial resolution and accuracy. SAR can also penetrate through clouds, vegetation, and other obstacles, which makes it ideal for applications such as mapping forest cover, monitoring glaciers and ice sheets, and tracking ocean waves and currents.

### Other Satellite Aspects to Know

Although I said that images are preprocessed if there are stitched together they also need a few things and the first one is illumination. Illumination refers to the amount and direction of light that is falling on the Earth's surface when a satellite image is captured. The illumination conditions at the time of image capture can affect the appearance of features on the image, particularly those with a high relief or slope.


Next aspect you need to understand is orthorectification. This is the process of removing geometric distortions from an image caused by the Earth's curvature, terrain relief, and sensor perspective. It involves reprojecting the image onto a flat surface, using ground control points to correct for distortions.

<p align = "center">
<img src="/images/posts/geospatial_1_orthorectification.png" width="400">
</p>
<p align = "center">
Figure 6 Orthorectification: the process of removing geometric distortions from an image caused by the Earth's curvature .
</p>

Lastly is good to know geographical metadata that often is provided so what is azimuth and altitude. Azimuth: This is the angle between true north and the direction to a point of interest and altitude refers to the orientation of a satellite or sensor relative to the Earth's surface. Those information with geographical positioning are major information used in satellite images from geography point of view.

<p align = "center">
<img src="/images/posts/geospatial_1_azimuth.svg" width="250">
</p>
<p align = "center">
Figure 7 The azimuth is the angle formed between a reference direction (in this example north) and a line from the observer to a point of interest projected on the same plane as the reference direction orthogonal to the zenith.
</p>

### UAV Images

Satellite images are usually well-prepared, but images from drones and planes can be more challenging to work with. These images might not be taken from a straight-down perspective (known as NADIR format), and they might not be in the same coordinate system as maps. To make these images usable, we need to re-project them and stitch them together.

<p align = "center">
<img src="/images/posts/geospatial_1_stich.webp" width="450">
</p>
<p align = "center">
Figure 8 process of stitching images taken from drone to one image.
</p>

#### UAV image pre and post processing

Before we can analyze images from UAVs, we need to prepare them by stitching them together to remove unnecessary data and create a seamless image. This process is known as [Orthomosaic Generation](https://pro.arcgis.com/en/pro-app/latest/help/data/imagery/generate-an-orthomosaics-using-the-orthomosaic-wizard.htm) and is also used in [SLAM (Simultaneous Localization and Mapping](https://www.mathworks.com/discovery/slam.html). The process involves several steps, including image calibration, feature detection, feature matching, global alignment, and blending.

Image calibration :wrench: -> Feature detection :mag: -> Feature matching (translation estimation) :couple: -> Global alignment :straight_ruler: -> Blending :link:

Now let's discuse each step shortly. Image calibration :wrench: is the process involves correcting the distortions that may be present in the UAV images due to lens aberration, sensor misalignment, or atmospheric effects. Image calibration includes operations such as radiometric correction, geometric correction, and color balancing to ensure that the images are accurate and consistent. Sample in opencv for [image calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) or more precisle [article](https://www.researchgate.net/publication/274674914_Digital_Camera_Calibration_Using_Images_Taken_From_An_Unmanned_Aerial_Vehicle) about it for UAV.

Feature detection :mag: is the process of identifying important features or points of interest in UAV images. This can include things like corners, edges, and distinct shapes. Common feature detection techniques include corner detection, edge detection, and blob detection. Some common algorithms used for feature detection are SIRF, SURF, and ORB ([medium paper :notebook:](https://mikhail-kennerley.medium.com/a-comparison-of-sift-surf-and-orb-on-opencv-59119b9ec3d0) comparing those methods).

Once the features are detected in the UAV images, they are feature matched :couple: across multiple images to determine their corresponding positions. Feature matching involves computing the similarity between the features in different images and estimating the translation between the images and often often is used [RANSAC algorithm](http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf).

Global alignment :straight_ruler: is the process of aligning all the UAV images to a common coordinate system, so they can be accurately compared and analyzed. This is done using ground control points (GCPs) or other reference data. The global alignment process accounts for any differences in orientation, scale, or perspective between the images, to create a seamless and accurate image.

<p align = "center">
<img src="/images/posts/geospatial_1_blending.webp" width="400">
</p>
<p align = "center">
Figure 9 Image blending where apple should be the left part of the final image and orange in the right part.
</p>

Blending :link:: The final step in UAV image processing is to blend the aligned images into a seamless mosaic or orthomosaic. This involves adjusting the brightness and color balance of the images to create a visually appealing and geographically accurate representation of the UAV survey area. Image Blending often uses [Pyramids](https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html) for this task.

However I don't have much of the experience in this field so I don't know precisly what steps are still needed or what steps are dealt by data scientis or data engieenr. I think that UAV images are often already calibrated and provided with GPS data so calibration and alligmnet becomes obsolite. However Featrue detection, matching and blending can still needed to be done however how hard it is it can depend on what information UAV is provided to the image.

In the context of data pipeline I mostly come across with dealing with this task with serveless function such with aws lambda functions and EC2 instances. However to deal with it I would start with especting this open source toolkit for it [OpenDroneMap](https://www.opendronemap.org) and then maybe look across stories look for similar stories in the internet and tools for it.

## Satellite data structures

Before I finish the first part of this blog, which mostly focuses on concepts in satellite and UAV images, I want to discuss data structures used with satellite data, specifically how this data is stored and read.

Firstly, before focusing on satellite images, I need to mention that geospatial data is mostly stored in two formats: shapes and rasters. Rasters, are just images of the earth with various spectral resolutions, not only RGB. However, if you go to OpenStreetMaps, you will mainly see some shapes and not images because it is less memory-consuming to store shapes such as lines, points, or polygons. We can often refer to the earth image through a Coordinate Reference System (CRS), which is a method used to describe and locate features on the surface of the Earth. Thus sometimes we store data as shapes and not only rasters.

### Vector spatial structures

Shapes are often stored in [shapefiles](https://desktop.arcgis.com/en/arcmap/latest/manage-data/shapefiles/what-is-a-shapefile.htm) (.shp|.shx|.dbf) format. This format was developed by the Environmental Systems Research Institute (ESRI) and is now widely used by many GIS software applications. The structure of a Shapefile is organized into three main components: header, record, and geometry. The header contains information about the file, including the shape type, bounding box, and projection information. The record describes the attributes associated with a feature, such as the name, population, or elevation. The geometry contains the spatial location and shape of the feature, such as a point or a polygon.

<p align = "center">
<img src="/images/posts/geospatial_1_vector.png" width="400">
</p>
<p align = "center">
Figure 10 Open Street Map of my town. It is mostly constructed with vector structures such as lines and polygons.
</p>

The second widely used format is [GeoJSON](https://libgeos.org/specifications/geojson/), which is based on the popular file format JSON. It is a lightweight, text-based format designed to represent simple or complex geometries, including points, lines, polygons, and multi-geometries, along with their associated properties. It is supported by many GIS software applications and web mapping platforms, such as Google Maps and Leaflet.

Comparing the two formats, I would choose GeoJSON formats due to a few intakes. Shapefiles consist of multiple files, including a main file for geometry, an index file, and a database file for attributes. In contrast, GeoJSON is a single file format that includes both the geometry and attributes in one file. **GeoJSON is faster**. Shapefiles support a wide range of geometry types, including points, lines, and polygons, while GeoJSON also supports these geometry types as well as more complex geometries like multipoints, multilines, and multipolygons. GeoJSON has the advantage of being more flexible and easier to use with web applications.

There also exist [Well-Known Text (WKT)](https://libgeos.org/specifications/wkt/) and [Well-Known Binary (WKB)](https://libgeos.org/specifications/wkb/), common text-based and binary-based formats, respectively, for representing spatial data in a standardized format. In terms of usage, WKT and WKB are primarily used for exchanging spatial data between different software applications, while GeoJSON and Shapefile are commonly used for storing and sharing spatial data within GIS applications.

Summarizing, GeoJSON formats are the most commonly used formats as they are in JSON format, which is widely used and developed across many developer fields. Shapefile is older, however, still used, and WKT and WKB are simple structures that are great to use when using different software applications. Nowadays, the most popular libraries and frameworks primarily try to support GeoJSON.

### Raster spatial structures

The second type of data structures used in Earth Observation (EO) are raster data, which are simply a pixel grid representation (similar to the photos taken by an iPhone). The term "raster" is used instead of images refers to the fact that normal images have a spatial resolution of RGB, whereas satellite images are not bound to only this resolution.

<p align = "center">
<img src="/images/posts/geospatial_1_raster.png" width="500">
</p>
<p align = "center">
Figure 11 Satellite raster data.
</p>

Raster data is typically stored in two formats: TIFF and JPEG2000. [TIFF(Tagged Image File Format)](https://en.wikipedia.org/wiki/TIFF) is a widely used raster image format for storing and exchanging digital images. It is a flexible format that can store a variety of data types, including color and various spatial resolutions, and supports lossless compression. [GeoTIFF](https://www.gislounge.com/what-is-a-geotiff/) is a variant of the TIFF format that includes geospatial metadata, such as [coordinate reference system (CRS)](https://geopandas.org/en/stable/docs/user_guide/projections.html) information and projection parameters.

[JPEG2000](https://en.wikipedia.org/wiki/JPEG_2000) is a newer raster image format that was designed to overcome some of the limitations of the JPEG format. It uses wavelet compression, which allows for higher compression ratios with less loss of image quality compared to JPEG. JPEG2000 is less common than TIFF, and is mainly used with raster data that are image-like, such as RGB format, rather than like TIFF with every spatial resolution.

What I love about the geospatial community is their commitment to open-source and cloud-based technologies. The challenge of storing and utilizing whole Earth raster images has led to the development of more cloud-specific formats, such as Zarr, COG, and netCDF.

<p align = "center">
<img src="/images/posts/geospatial_1_cloud.png" width="400">
</p>
<p align = "center">
Figure 12 Ah cloud computing and big data.
</p>

[NetCDF (network Common Data Form)](https://pro.arcgis.com/en/pro-app/latest/help/data/multidimensional/what-is-netcdf-data.htm) is a file format and data model for storing and sharing scientific data in a platform-independent and self-describing way. The netCDF data model is based on a hierarchical structure of dimensions, variables, and attributes. A netCDF file can contain one or more variables, each of which has a name, data type, and dimensions that define its shape. Variables can be multidimensional, and each dimension can have a name and a size.

[Zarr](https://zarr.dev) is a format for storing array data that is designed for efficient storage and retrieval of large, multi-dimensional arrays. It supports a range of compression and encoding options to optimize data storage, and can be easily chunked and distributed across multiple nodes for parallel processing. Zarr is often used for storing large, multidimensional arrays of satellite imagery or other geospatial data, and is supported by a range of open-source libraries and cloud storage services.

[COG (Cloud-Optimized GeoTIFF)](https://www.cogeo.org), on the other hand, is a format for storing geospatial raster data in a way that enables efficient access and processing in cloud environments. COG files are regular GeoTIFF files that have been structured to enable efficient access to specific subsets of the data without the need to read the entire file. COG files can be efficiently streamed over the internet, and can be easily accessed and processed using a range of open-source tools and cloud services. COG files are often used for storing and distributing large collections of geospatial data, such as satellite imagery or topographic maps.

<p align = "center">
<img src="/images/posts/geospatial_1_decide.jpg" width="300">
</p>
<p align = "center">
Figure 13 What to choose.
</p>

In general, netCDF is best suited for storing and working with large scientific datasets that require multidimensional arrays of numerical data, such as climate model output and oceanographic data. While the other mentioned formats are widely used for raster data, I have witnessed that COGs are more commonly used than zarr in the geospatial community. However, in developing communities, zarr is becoming increasingly popular for big data geoscience, such as those using [Pangeo](https://pangeo.io/index.html). As both formats are well-suited for big data cloud reading, I would choose zarr because it is not specific to spatial data. Therefore, as a globally used format, it will be better maintained and developed. However, COG format may be much better suited since it is developed primarily with a focus on raster images.

## STAC

Finally, I want to discuss what STAC is. [STAC (SpatioTemporal Asset Catalog)](https://stacspec.org/en) is a community-driven specification for organizing and sharing geospatial data as standardized, machine-readable metadata. It is designed to provide a uniform way of describing spatiotemporal data assets, such as satellite imagery, aerial photography, or other geospatial datasets, so they can be easily searched, discovered, and analyzed.

STAC provides a flexible metadata structure that can be used to describe a wide range of geospatial data assets, including their spatial and temporal coverage, resolution, format, quality, and other properties. With STAC metadata, users can enable various workflows, such as searching and discovering data assets, filtering and querying data based on specific criteria, and efficient access and processing of data in cloud environments.

STAC has gained popularity in the geospatial community as a way to standardize the way geospatial data is organized and shared. It makes it easier for users to discover and access the data they need. Many commercial and open-source geospatial tools and platforms have adopted STAC as a standard for organizing and sharing geospatial data. It can be used by government agencies, academic institutions, private companies, and non-profit organizations to manage and share their geospatial data.

In simpler terms, STAC is a hierarchical metadata structure that points to spatial data. It is widely used to analyze and find the necessary data without reading a large amount of data. It is lightweight and enables users to quickly pinpoint a range of SpatioTemporal data that they need, allowing them to load only the required data without the need for waiting hours for the data to load and use only a small percentage of it.

## In the next episode :dog2:

In the next blog post, I will focus on demonstrating the Python tools that are used in satellite data processing and visualization, along with their application in AI and machine learning. The focus will be more practical, with an emphasis on how it is done rather than covering everything. I will also provide some cool frameworks and sources to explore further.

In this blog I intended to give you my knowledge about satellite and UAV data, which are frequently used or referenced in the field. While it would be ideal for data scientists or ML engineers to gain a deeper understanding of these concepts, it is not necessary to be a geography expert or have an extensive knowledge of the field.

Based on this understanding, you can begin to explore the application of computer science and machine learning in this domain. It is crucial to understand the importance of processing big data in this field and maybe your task to help them as people without computer science knowledge may struggle to optimize it effectively. For Machine Learning researchers and engineers, I recommend exploring the following interesting research labs, publications, and works:

- [EcoVision](https://prs.igp.ethz.ch/ecovision.html)
- [ECEO](https://www.ga.gov.au/about/careers/graduate-program)
- [Geoscience Australia](https://www.ga.gov.au/about/careers/graduate-program)

I also recommend the following sources, as they provide more information on geospatial data experience for machine learning developers:

- [GeoAwesome](https://geoawesomeness.com): The world's largest geospatial community (They say so on their website), where geospatial experts and enthusiasts share their passion, knowledge, and expertise for all things geo. It features cool blogs and podcasts.
- [satellite-image-deep-learning](https://www.satellite-image-deep-learning.com): A newsletter that summarizes the latest developments in the domain of deep learning applied to satellite and aerial imagery. The author also creates a [course](https://github.com/satellite-image-deep-learning/course) in this domain.
- [STAC example](https://stacspec.org/en/tutorials/access-sentinel-2-data-aws/) of working with Sentinel-2 data from AWS.
- [Workshop](https://geohackweek.github.io/raster/04-workingwithrasters/)  on raster processing using python tools.
- [STAC Best Practices](https://github.com/radiantearth/stac-spec/blob/master/best-practices.md) official readme.
- [Radiant Earth Foundation](https://radiant.earth): An American non-profit organization founded in 2016, whose goal is to apply machine learning for Earth observation to meet the Sustainable Development Goals. They support the STAC project and other interesting projects.
- [Pangeo](https://pangeo.io/meeting-notes.html): A community platform for Big Data geoscience. Promotes popular Python geoframeworks with a modern approach to community and tool development/support.
- [Picterra](https://picterra.ch): The "leading" platform for geospatial MLOps. They have great blogs, and you can check how MLOps looks for geospatial problems.