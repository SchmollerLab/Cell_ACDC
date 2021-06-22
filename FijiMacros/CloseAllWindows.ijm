// "Close All Windows"
// This macro closes all image windows without
// displaying "save changes" dialog boxes, i.e.
// all unsaved changes are lost. To create a 
// "Close All Windows" command, add it to the
// StartupMacros file, or drop it into the
// ImageJ/plugins/Macros folder.
// Note that ImageJ 1.37 has a bug that
// causes this macro to run very slowly.

  macro "Close All Windows" { 
      while (nImages>0) { 
          selectImage(nImages); 
          close(); 
      } 
  } 