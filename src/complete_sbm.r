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
thresh <- read.csv("data-samples/manual/complete_thresholds2.csv", row.names = "X")
files <- list.files("data-samples/graphs/tab_nodes")
# pb <- txtProgressBar(min = 0, max = length(files), initial = 0)
for (i  in 1:length(files)){
    f <- files[i]
    for (chars in c("p01","p06","p10","p35")){
        # print(paste(i,"/",length(files)))
        if (grepl(chars, f, fixed = TRUE)&!(grepl(chars, "ini", fixed = TRUE))){
            name <- paste("data-samples/graphs/tab_nodes/", f, sep = "")
            x <- unlist(strsplit(f, "_"))
            y <- substr(x[2], 2, 5)
            p <- substr(x[3], 2, 3)
            g <- read.gexf(name)
            g_ig <- gexf.to.igraph(g)
            bin_mat <- as(as.matrix((as_adj(g_ig, attr = "weight") >= thresh[p, paste("X", y, sep = "")])), "dgCMatrix") # nolint
            sol_sbm <- greed(bin_mat, model = Sbm(alpha = 0.5, a0 = 1, b0 = 1, type = "directed"), K = 2, verbose = FALSE) # nolint
            sol_dc <- greed(bin_mat, model = DcSbm(alpha = 0.1, type = "directed"), K = 2, verbose = FALSE) # nolint
            v <- V(g_ig)$name
            c1 <- clustering(sol_sbm)
            c2 <- clustering(sol_dc)
            m1 <- modularity(g_ig, c1)
            m2 <- modularity(g_ig, c2)
            icl1 <- ICL(sol_sbm)
            icl2 <- ICL(sol_dc)

            print(K(sol_sbm))
            print(m1)
            print(icl1)
            print(K(sol_dc))
            print(m2)
            print(icl2)

            s1 <- c("country,block,modularity,icl")
            s2 <- c("country,block,modularity,icl")
            for (i in 1:length(v)){
                s1 <- c(s1, paste(v[i], c1[i], m1, icl1, sep = ","))
                s2 <- c(s2, paste(v[i], c2[i], m2, icl2, sep = ","))
            }
            f_o <- unlist(strsplit(f, "[.]"))
            out_name1 <- paste("data-samples/sbm/v4/", f_o[1], "_sbm_n4g.csv", sep = "") # nolint
            out_name2 <- paste("data-samples/sbm/v4/", f_o[1], "_dc_n4g.csv", sep = "") # nolint
            fileConn1 <- file(out_name1)
            fileConn2 <- file(out_name2)
            writeLines(s1, fileConn1)
            writeLines(s2, fileConn2)
            close(fileConn1)
            close(fileConn2)
        }
    }
}
# close(pb)