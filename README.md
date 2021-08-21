# Decoding and rendering MBTiles files using Python

Python scripts for experimenting with `MBTiles` files.
Mostly to see how complex rendering is and how feasible
it would be to implement this on a embedded processor.

In the below images zoom levels 1-14 are supported by the `MBTiles` file.
Zoom levels 15-18 done just by scaling the vector image.

![gdansk_gif](gdansk/gdansk.gif)

![gdansk_z12](gdansk/012.png)
![gdansk_z14](gdansk/014.png)

![oliwa_gif](oliwa/oliwa.gif)

![oliwa_z12](oliwa/012.png)
![oliwa_z14](oliwa/014.png)


# TODO

 - the way the polygons are encoded ([link](https://docs.mapbox.com/vector-tiles/specification/#winding-order))
   makes them slow to draw in `PIL` (drawing on separete canvas and pasting)
 - there is no 'smart' detection of objects out of frame, so drawing with
   big tile sizes is slow

