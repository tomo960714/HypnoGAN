# Step-by-step guide of how to download data from NSRR and gather sleeping stage information from the data

1. Download dataset from NSRR into a folder. To make this process work, EDFs and xml files are needed. (Download link available at NSRR)
2. Run DOCKER and connect to the server and mount folder to Luna's already prepared data folder with the following command:

```
docker run --rm -it -v Data_folder:/data remnrem/luna /bin/bash

```
3. Create a 'sample list'.
```
luna --build Data_folder/../edfs Data_folder/.../annotations-events-nsrr/ -ext=-nsrr.xml > s.lst
```
4. Run STAGES command using the sample list to get sleeping stage from all file
```
luna s.lst -t o1 -s STAGE
```
&emsp; Run the following code to only 
```
luna s.lst 1 5 -t o1 -s STAGE
```

5. Process the folders containing the resulted files with the python script. (In progress)
