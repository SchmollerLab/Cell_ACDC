//Channels names
//channels = newArray("mNeon","mKate","phase_contr");
channels = newArray("phase_contr","mCitrine");

macro_path = File.directory();

//File dialog and open
id = File.openDialog("Select a czi file");

setBatchMode(true); //when setBatchMode is set to true imageJ does not display the images. This speeds up the process by 20x.
run("Bio-Formats Macro Extensions");

// Initialize file
Ext.setId(id);

//Get file information
Ext.getSeriesCount(seriesCount);
name = File.getName(id);
print("Number of series in "+name+" is: "+seriesCount);

//Create folder
path = File.getParent(id);
wpath= replace(path, "/", "\\");
osInfo = getInfo("os.name");
WindowsIdx = indexOf(osInfo, "Windows");
if (WindowsIdx != -1)
{
	exec("cmd /c C:\\Windows\\explorer.exe \""+ wpath +"\"");
} else {
	exec("open " + path);
}
folder = path+"/TIFFs";
MIA = folder;
File.makeDirectory(MIA);

//open each series by splitting channels and saving them separately into .tif files
S = 1; //start from S series (from 1)
End = seriesCount; //S for 1 for loop iteration, seriesCount for all series
print("Number of positions: "+seriesCount);
for (s=S-1; s<End; s++) { //for loop for iterating through the series
	seriesNum = s+1;
	run("Bio-Formats Importer", "open=["+id+"] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT use_virtual_stack series_"+seriesNum);
	//run("Bio-Formats Importer", "open=["+id+"] color_mode=Default rois_import=[ROI manager] split_focal split_timepoints view=Hyperstack stack_order=XYCZT use_virtual_stack series_"+(s+1));
	nameWithExt = File.getName(id);

	name = File.nameWithoutExtension;	
	Ext.setSeries(s); //Sets the current series within the active dataset.
	Ext.getSizeC(sizeC); //Gets the number of channels in the current series.
	//Ext.getSizeZ(sizeZ); //Gets the number of focal planes in the current series.
	print("Saving s="+seriesNum+"/"+End+"..."); //Display message
	pos_path = folder+"/Position_"+seriesNum;
	File.makeDirectory(pos_path);
	images_path = pos_path+"/Images";
	File.makeDirectory(images_path);
	C = 0;
	CEnd = sizeC;
	
	for (c=C; c<CEnd; c++) { //for loop for iterating through the channels
		selectImage(1);
		print("    Saving channel="+c+1+"/"+CEnd+"..."); //Display message
		scTif = images_path+"/"+name+"_s"+nss(seriesNum, seriesCount)+"_"+channels[c]+".tif";		
		saveAs("Tiff", scTif);
		close();
	}
}
print("Conversion to TIFFs finished!");


function nss(n, numPos){
	ss = "";
	if (n<10 && numPos>=10)
		ss = "0"+n;	
	else
		ss = ""+n;
	return ss;
}