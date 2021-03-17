//Channels names
channels = newArray("mNeon","mKate","phase_contr");

setBatchMode(true); //when setBatchMode is set to true imageJ does not display the images. This speeds up the process by 20x.
run("Bio-Formats Macro Extensions");

//Dialog folder and get file list
nd2_folder = getDirectory("Choose folder containing multiple ND2 files")
nd2_folder_name = File.getName(nd2_folder);
ids = getFileList(nd2_folder);
Array.sort(ids);

sEnd = ids.length //1 for just first file and ids.length for all files in the folder

// If .nd2 files are not in a ND2s subfolder create the folder and move files there
if (nd2_folder_name != "ND2s") {
	nd2_folder_new = nd2_folder + "ND2s\\";
	File.makeDirectory(nd2_folder_new);

	for (s = 0; s < sEnd; s++) {
		id = nd2_folder + ids[s];
		new_id = nd2_folder_new + ids[s];
		File.rename(id,new_id);
	}
	nd2_folder = nd2_folder_new;
}

//Create TIFFs folder
path = File.getParent(nd2_folder);
wpath= replace(path, "/", "\\");
exec("cmd /c C:\\Windows\\explorer.exe \""+ wpath +"\"");
TIFFs = path+"/TIFFs";
File.makeDirectory(TIFFs);



for (s = 0; s < sEnd; s++) {
	id = nd2_folder + ids[s];
	
	// Initialize file
	Ext.setId(id);
	
	//Get file name
	name = File.getName(id);
	
//	//Open nd2 files with Bio-Formats Importer
	seriesNum = s+1;
	run("Bio-Formats Importer", "open=["+id+"] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT use_virtual_stack series_"+seriesNum);
//	//run("Bio-Formats Importer", "open=["+id+"] color_mode=Default rois_import=[ROI manager] split_focal split_timepoints view=Hyperstack stack_order=XYCZT use_virtual_stack series_"+(s+1));
	name = File.nameWithoutExtension;
	Ext.getSizeC(sizeC); //Gets the number of channels in the current series.
	print("Saving s="+seriesNum+"/"+sEnd+"..."); //Display message
	pos_path = TIFFs+"/Position_"+seriesNum;
	File.makeDirectory(pos_path);
	images_path = pos_path+"/Images";
	File.makeDirectory(images_path);
	C = 0;
	CEnd = sizeC;
		
	for (c=C; c<CEnd; c++) { //for loop for iterating through the channels
		print("    Saving channel="+c+1+"/"+CEnd+"..."); //Display message
		scTif = images_path+"/"+name+"_s"+nss(seriesNum, 11)+"_"+channels[c]+".tif";		
		selectWindow(name+".nd2 - C="+c);
		saveAs("Tiff", scTif);
	}
	print("Saved!");
}

close_all_macro_path = "C:/MyPrograms/Fiji/MyMacros/CloseAllWindows.ijm";
runMacro(close_all_macro_path,true);

function nss(n, numPos){
	ss = "";
	if (n<10 && numPos>=10)
		ss = "0"+n;	
	else
		ss = ""+n;
	return ss;
}