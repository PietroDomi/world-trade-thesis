getwd()
setwd("H:/My Drive/COMEXT")
getwd()
library(future, quietly = TRUE) # allows parralel processing in greed()
library(Matrix, quietly = TRUE) # sparse matrix
library(ggplot2, quietly = TRUE) # ploting and data
library(greed, quietly = TRUE)
library(igraph, quietly = TRUE)
library(dplyr, quietly = TRUE)
#library(ggpubr, quietly = TRUE)
library(rgexf)

future::plan("multisession", workers = 5) # may be increased
thresh <- read.csv("data-samples/manual/complete_thresholds.csv", row.names = "X")
files <- list.files("data-samples/graphs/complete")
# pb <- txtProgressBar(min = 0, max = length(files), initial = 0)
for (i  in 1:length(files)){
    chars = ""
    f <- files[i]
    print(paste(i,"/",length(files)))
    if (grepl(chars, f, fixed = TRUE)&!(grepl(chars, "ini", fixed = TRUE))){
        name <- paste("data-samples/graphs/complete/", f, sep = "")
        x <- unlist(strsplit(f, "_"))
        y <- substr(x[2], 2, 5)
        p <- substr(x[3], 2, 3)
        g <- read.gexf(name)
        g_ig <- gexf.to.igraph(g)
        bin_mat <- as(as.matrix((as_adj(g_ig,attr="weight") >= thresh[p,paste("X",y,sep="")])),"dgCMatrix")
        sol <- greed(bin_mat, model = Sbm(alpha = 1, a0 = 5, b0 = 2, type = "directed"), K = 2, verbose = FALSE)
        v <- V(g_ig)$name
        c <- clustering(sol)
        s <- c("country,block")
        for (i in 1:length(v)){
            s <- c(s,paste(v[i],c[i],sep=","))
        }
        f_o <- unlist(strsplit(f, "[.]"))
        out_name <- paste("data-samples/sbm/v2/", f_o[1], ".csv", sep = "")
        fileConn <- file(out_name)
        writeLines(s, fileConn)
        close(fileConn)
    }
}
# close(pb)