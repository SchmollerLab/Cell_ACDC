requires("1.52u");

setBatchMode(true); //when setBatchMode is set to true imageJ does not display the images. This speeds up the process by 20x.
run("Bio-Formats Macro Extensions");

//Channels names
channels = newArray(...);

//Dialog folder and get file list
id = ...;
dst_folderpath = ...;
files_folder = id;
files_folder_name = File.getName(files_folder);
ids = getFileList(files_folder);
natural_order = extract_digits(ids);
Array.sort(natural_order, ids);

sEnd = ids.length //1 for just first file and ids.length for all files in the folder

//Create TIFFs folder
path = dst_folderpath;
wpath= replace(path, "/", "\\");
osInfo = getInfo("os.name");
WindowsIdx = indexOf(osInfo, "Windows");
if (WindowsIdx != -1)
 {
	exec("cmd /c C:\\Windows\\explorer.exe \""+ wpath +"\"");
} else {
	exec("open " + path);
}

for (s = 0; s < sEnd; s++) {
	id = files_folder + "/" + ids[s];
	print(id);
	
	// Initialize file
	Ext.setId(id);
	
	//Get file name
	nameWithExt = File.getName(id);
	
//	//Open czi files with Bio-Formats Importer
	seriesNum = s+1;
	run("Bio-Formats Importer", "open=["+id+"] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT use_virtual_stack series_"+seriesNum);
//	//run("Bio-Formats Importer", "open=["+id+"] color_mode=Default rois_import=[ROI manager] split_focal split_timepoints view=Hyperstack stack_order=XYCZT use_virtual_stack series_"+(s+1));
	filenameNoExtension = File.nameWithoutExtension;
	Ext.getSizeC(sizeC); //Gets the number of channels in the current series.
	print("Saving s="+seriesNum+"/"+sEnd+"..."); //Display message

	C = 0;
	CEnd = sizeC;

	// Create metadata file with basename
    pos_num = nss(seriesNum, sEnd);
    print("Creating metadata file for Position_" + seriesNum);
    metadata_filepath = dst_folderpath + "/" + filenameNoExtension + "_metadata.csv";
    
    metadata_file = File.open(metadata_filepath);
    metadata_str = "Description,values";
    
     // Add sizes to metadata
    Ext.getSizeT(sizeT);
   	metadata_str = metadata_str + "\n" + "SizeT," + sizeT;
    Ext.getSizeZ(sizeZ);
    metadata_str = metadata_str + "\n" + "SizeZ," + sizeZ;
    
    // Add physical sizes to metadata file
    Ext.getPixelsPhysicalSizeZ(physicalSizeZ);
    metadata_str = metadata_str + "\n" + "PhysicalSizeZ," + physicalSizeZ;
    Ext.getPixelsPhysicalSizeY(physicalSizeY);
    metadata_str = metadata_str + "\n" + "PhysicalSizeY," + physicalSizeY;
    
    if (sizeZ > 1) {
    	Ext.getPixelsPhysicalSizeX(physicalSizeX);
    	metadata_str = metadata_str + "\n" + "PhysicalSizeX," + physicalSizeX;
    } else {
    	print("Image is 2D");
    }
    
    
    if (sizeT > 1) {
    	Ext.getPixelsTimeIncrement(timeIncrement);
    	metadata_str = metadata_str + "\n" + "TimeIncrement," + timeIncrement;
    } else {
    	print("Image file is not timelapse");
    }
    
	scTif = dst_folderpath + "/" + filenameNoExtension + ".tif";		
	selectImage(1);
	
	print("---------");
	print(scTif);
	print("---------");
	
	saveAs("Tiff", scTif);
	close();

	print(metadata_file, metadata_str);
	File.close(metadata_file);
	print("Saved!");
}

print("All done!");
// run("Quit");

function nss(n, numPos){
	ss = "";
	if (n<10 && numPos>=10)
		ss = "0"+n;	
	else
		ss = ""+n;
	return ss;
}

function extract_digits(a) {
	arr2 = newArray; //return array containing digits
	for (i = 0; i < a.length; i++) {
		str = a[i];
		digits = "";
		len = lengthOf(str);
		for (j = 0; j < len; j++) {
			ch = substring(str, j, j+1);
			num = parseInt(ch);
			if(!isNaN(parseInt(ch)))
				digits += ch;
		}
		arr2 = Array.concat(arr2, parseInt(digits));
	}
	return arr2;
}