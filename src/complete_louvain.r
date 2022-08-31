getwd()
setwd("H:/My Drive/COMEXT")
getwd()
library(future, quietly = TRUE) # allows parralel processing in greed()
library(Matrix, quietly = TRUE) # sparse matrix
library(ggplot2, quietly = TRUE) # ploting and data
library(greed, quietly = TRUE)
library(igraph, quietly = TRUE)
library(dplyr, quietly = TRUE)
library(rgexf)

future::plan("multisession", workers = 5) # may be increased
files <- list.files("data-samples/graphs/complete")
for (i in 1:length(files)){
    f <- files[i]
    for (chars in c("p01","p10","p06","p35")){
        if (grepl(chars, f, fixed = TRUE)&!(grepl("ini", f, fixed = TRUE))){
            name <- paste("data-samples/graphs/complete/", f, sep = "")
            x <- unlist(strsplit(f, "_"))
            y <- substr(x[2], 2, 5)
            p <- substr(x[3], 2, 3)
            g <- read.gexf(name)
            g_ig <- gexf.to.igraph(g)

            c_lou <- cluster_louvain(g_ig, resolution = 0.5)
            mod <- modularity(c_lou)
            mem <- membership(c_lou)
            print(f)
            print(length(g_ig))
            print(length(c_lou))
            print(mod)
            v <- V(g_ig)$name
            s <- c("country,block,modularity")
            for (j in 1:length(v)){
                s <- c(s, paste(v[j], mem[j], mod, sep = ","))
            }
            f_o <- unlist(strsplit(f, "[.]"))
            out_name <- paste("data-samples/louvain/v1/", f_o[1], "_r05.csv", sep = "") # nolint
            fileConn <- file(out_name)
            writeLines(s, fileConn)
            close(fileConn)
        }
    }
}