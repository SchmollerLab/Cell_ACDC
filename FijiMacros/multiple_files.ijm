requires("1.52u");

//Channels names
channels = newArray("mNeon","mKate","phase_contr");
// channels = newArray("AlexaFluor","phase_contr","DAPI","Cy3");

setBatchMode(true); //when setBatchMode is set to true imageJ does not display the images. This speeds up the process by 20x.
run("Bio-Formats Macro Extensions");

//Dialog folder and get file list
czi_folder = getDirectory("Choose folder containing multiple microscopy files")
czi_folder_name = File.getName(czi_folder);
ids = getFileList(czi_folder);
natural_order = extract_digits(ids);
Array.sort(natural_order, ids);

sEnd = ids.length //1 for just first file and ids.length for all files in the folder

// If .czi files are not in a CZIs subfolder create the folder and move files there
if (czi_folder_name != "raw_microscopy_files") {
	czi_folder_new = czi_folder + "raw_microscopy_files";
	File.makeDirectory(czi_folder_new);

	for (s = 0; s < sEnd; s++) {
		id = czi_folder + ids[s];
		new_id = czi_folder_new + "/" + ids[s];
		File.rename(id,new_id);
	}
	czi_folder = czi_folder_new;
}

//Create TIFFs folder
path = File.getParent(czi_folder);
wpath= replace(path, "/", "\\");
osInfo = getInfo("os.name");
WindowsIdx = indexOf(osInfo, "Windows");
if (WindowsIdx != -1)
 {
	exec("cmd /c C:\\Windows\\explorer.exe \""+ wpath +"\"");
} else {
	exec("open " + path);
}

TIFFs = path;
File.makeDirectory(TIFFs);

for (s = 0; s < sEnd; s++) {
	id = czi_folder + "/" + ids[s];
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
	pos_path = TIFFs+"/Position_"+seriesNum;
	File.makeDirectory(pos_path);
	images_path = pos_path+"/Images";
	File.makeDirectory(images_path);
	C = 0;
	CEnd = sizeC;

	// Create metadata file with basename
    pos_num = nss(seriesNum, sEnd);
    print("Creating metadata file for Position_" + seriesNum);
    basename_string = filenameNoExtension + "_s" + pos_num + "_";
    metadata_filepath = images_path + "/" + basename_string + "metadata.csv";
    metadata_file = File.open(metadata_filepath);
    metadata_str = "Description,values\nbasename," + basename_string;
    
    // Add sizes to metadata
    Ext.getSizeT(sizeT);
   	metadata_str = metadata_str + "\n" + "SizeT," + sizeT;
    Ext.getSizeZ(sizeZ);
    metadata_str = metadata_str + "\n" + "SizeZ," + sizeZ;
    
    // Add physical sizes to metadata file
    Ext.getPixelsPhysicalSizeX(physicalSizeX);
    metadata_str = metadata_str + "\n" + "PhysicalSizeX," + physicalSizeX;
    Ext.getPixelsPhysicalSizeY(physicalSizeY);
    metadata_str = metadata_str + "\n" + "PhysicalSizeY," + physicalSizeY;
    
    if (sizeZ > 1) {
    	Ext.getPixelsPhysicalSizeZ(physicalSizeZ);
    	metadata_str = metadata_str + "\n" + "PhysicalSizeZ," + physicalSizeZ;
    } else {
    	print("Image is 2D");
    }
    
	for (c=C; c<CEnd; c++) { //for loop for iterating through the channels
		print("    Saving channel="+c+1+"/"+CEnd+"..."); //Display message
		scTif = images_path + "/" + basename_string + channels[c] + ".tif";		
		selectImage(1);
		saveAs("Tiff", scTif);
		close();
		channel_desc = "channel_" + c + "_name";
		metadata_str = metadata_str + "\n" + channel_desc + "," + channels[c];
	}
	print(metadata_file, metadata_str);
	File.close(metadata_file);
	print("Saved!");
}

print("All done!");

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