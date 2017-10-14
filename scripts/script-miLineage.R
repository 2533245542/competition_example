# This script runs on windows and -nix environment. It is called in shiny server function; the script copy miLineageResult folder 
# to the current working directory, assume miLineage will create the needed .xml and .txt documents, and run graphlan 
# and then generate a .png file and store it in object miPlot. So what the server needs to do is just to source() this
# script and run grid::grid.raster(miPlot)

#Assume user already installed miLineage.tar ***CRAN upload can solve this*** (also install mppa, geepack, data.table)
#Assuem user already installed biopython
#Assume user already installed png ***
#Assume user already installed pip
library(png)
library(miLineage)

Windows <- Sys.info()[['sysname']] == "Windows"
Mac <- Sys.info()[['sysname']] == "Darwin"
Linux <- Sys.info()[['sysname']] == "Linux"

if (Windows) {
  # ***miLineage is working, outputting .xml, annot.txtâ¦ â¦***
  # ***should output to paste0(getwd(), â/milineagAnalysis/milineage_resultâ)***
  # ***name of the output should be stored in variables***
  # ***e.g:    tree_file = âhmptree.xmlâ, newTree_file = ânewhmptree.xmlâ, annot_file = âannot_txtâ, ***
  # *** picture_file = âpic.pngâ, dpi_para = 12, size_para = 11 ***
  
  # Assume we already got files needed for running graphlan
  
  #set python path
  pyPath <- "set PATH=%PATH%;C:/Python27"
  
  #set PATH
  curDir <- getwd()
  pathDir <- "/milineageResult/milineage_graphlan"
  sysPath <- paste0(";", curDir, pathDir)
  
  #workout the picture
  
  runAnnotate <-
    paste(
      "python.exe",
      paste0(
        getwd(),
        "/milineageResult/milineage_graphlan/graphlan_annotate.py"
      ),
      "hmptree.xml",
      "hmptree.annot.xml",
      "--annot",
      "annot.txt"
    )
  
  runGraphlan <-
    paste(
      "python.exe",
      paste0(
        getwd(),
        "/milineageResult/milineage_graphlan/graphlan.py"
      ),
      "hmptree.annot.xml",
      "hmptree.png",
      "--dpi",
      150,
      "--size",
      14
    )
  
  cd <- paste0("cd ", getwd(), "/milineageResult")
  
  command1 <-
    paste(paste0(pyPath, sysPath), "&&", cd, "&&", runAnnotate)
  
  command2 <-
    paste(paste0(pyPath, sysPath), "&&", cd, "&&", runGraphlan)
  shell(command1)
  shell(command2)
  # output picture and display
  # pp <- readPNG(pasteins0("milineageResult/", picture_file))
  miPlot <- readPNG(paste0("milineageResult/", "hmptree.png"))
}

if (Mac) {
  # ***miLineage is working, outputting .xml, annot.txtâ¦ â¦***
  # ***should output to paste0(getwd(), â/milineagAnalysis/milineage_resultâ)***
  # ***name of the output should be stored in variables***
  # ***e.g:    tree_file = âhmptree.xmlâ, newTree_file = ânewhmptree.xmlâ, annot_file = âannot_txtâ, ***
  # *** picture_file = âpic.pngâ, dpi_para = 12, size_para = 11 ***
  
  # Assume we already got files needed for running graphlan
  
  # set PATH
  path <-
    paste0("export PATH=$PATH:",
           getwd(),
           "/milineageResult/milineage_graphlan")
  
  # work out picture
  cd <- paste0("cd ", getwd(), "/milineageResult")
  piece1 <-
    paste(
      "graphlan_annotate.py",
      "hmptree.xml",
      "hmptree.annot.xml",
      "--annot",
      "annot.txt"
    )
  piece2 <-
    paste("graphlan.py",
          "hmptree.annot.xml",
          "hmptree.png",
          "--dpi",
          150,
          "--size",
          14)
  line1 <- paste("(", cd, "&&", piece1, ")")
  line2 <- paste("(", cd, "&&", piece2, ")")
  
  # system(paste("graphlan_annotate.py", tree_file, newTree_file, "-- annot", annot_file))
  # piece2 <- paste("graphlan.py", newTree_file, picture_file, "âdpi", dpi_para, "âsize", size_para)
  command1 = paste("(", path , "&&", line1, ")")
  command2 = paste("(", path , "&&", line2, ")")
  system(command1)
  system(command2)
  # output picture and display
  # pp <- readPNG(pasteins0("milineageResult/", picture_file))
  miPlot <- readPNG(paste0("milineageResult/", "hmptree.png"))
}
if (Linux) {
  # ***miLineage is working, outputting .xml, annot.txtâ¦ â¦***
  # ***should output to paste0(getwd(), â/milineagAnalysis/milineage_resultâ)***
  # ***name of the output should be stored in variables***
  # ***e.g:    tree_file = âhmptree.xmlâ, newTree_file = ânewhmptree.xmlâ, annot_file = âannot_txtâ, ***
  # *** picture_file = âpic.pngâ, dpi_para = 12, size_para = 11 ***
  
  # Assume we already got files needed for running graphlan
  
  # set PATH
  path <-
    paste0("export PATH=$PATH:",
           getwd(),
           "/milineageResult/milineage_graphlan")
  
  # work out picture
  cd <- paste0("cd ", getwd(), "/milineageResult")
  piece1 <-
    paste(
      "graphlan_annotate.py",
      "hmptree.xml",
      "hmptree.annot.xml",
      "--annot",
      "annot.txt"
    )
  piece2 <-
    paste("graphlan.py",
          "hmptree.annot.xml",
          "hmptree.png",
          "--dpi",
          150,
          "--size",
          14)
  line1 <- paste("(", cd, "&&", piece1, ")")
  line2 <- paste("(", cd, "&&", piece2, ")")
  
  # system(paste("graphlan_annotate.py", tree_file, newTree_file, "-- annot", annot_file))
  # piece2 <- paste("graphlan.py", newTree_file, picture_file, "âdpi", dpi_para, "âsize", size_para)
  command1 = paste("(", path , "&&", line1, ")")
  command2 = paste("(", path , "&&", line2, ")")
  system(command1)
  system(command2)
  # output picture and display
  # pp <- readPNG(pasteins0("milineageResult/", picture_file))
  miPlot <- readPNG(paste0("milineageResult/", "hmptree.png"))
}
