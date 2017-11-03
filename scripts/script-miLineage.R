# This script runs on -nix environment. It is called in shiny server function; the script copy miLineageResult folder 
# to the current working directory, assume miLineage will create the needed .xml and .txt documents, and run graphlan 
# and then generate a .png file and store it in object miPlot. So what the server needs to do is just to source() this
# script and run grid::grid.raster(miPlot)

#Assume user already installed miLineage.tar ***CRAN upload can solve this*** (also install mppa, geepack, data.table)
#Assuem user already installed biopython
#Assume user already installed png ***
#Assume user already installed pip
library(png)
library(miLineage)
#move millineageAnaylysis folder from, package library, to, current working directory
# system(paste("cp -R", paste0(.libPaths(), "/miLineage/milineageResult"), paste0(getwd(), "/milineageResult")))
system(paste("cp -R", paste0(.libPaths(), "/miLineage/milineageResult"), paste0(getwd(), "/")))

  # ***miLineage is working, outputting .xml, annot.txt… …***
  # ***should output to paste0(getwd(), “/milineagAnalysis/milineage_result”)***
  # ***name of the output should be stored in variables***
  # ***e.g:    tree_file = “hmptree.xml”, newTree_file = “newhmptree.xml”, annot_file = “annot_txt”, ***
  # *** picture_file = “pic.png”, dpi_para = 12, size_para = 11 ***

# Assume we already got files needed for running graphlan

# set PATH
path <- paste0("export PATH=$PATH:",getwd(),"/milineageResult/milineage_graphlan")

# work out picture
cd <- paste0("cd ", getwd(), "/milineageResult")
piece1 <- paste("graphlan_annotate.py", "hmptree.xml", "hmptree.annot.xml", "--annot", "annot.txt")
piece2 <- paste("graphlan.py", "hmptree.annot.xml", "hmptree.png", "--dpi", 150, "--size", 14)
line1 <- paste("(", cd, "&&", piece1, ")")
line2 <- paste("(", cd, "&&", piece2, ")")

# system(paste("graphlan_annotate.py", tree_file, newTree_file, "-- annot", annot_file))
# piece2 <- paste("graphlan.py", newTree_file, picture_file, "—dpi", dpi_para, "—size", size_para)
command1 = paste("(", path , "&&", line1, ")")
command2 = paste("(", path , "&&", line2, ")")
system(command1)
system(command2)

# output picture and display
# pp <- readPNG(pasteins0("milineageResult/", picture_file))
miPlot <- readPNG(paste0("milineageResult/", "hmptree.png"))

