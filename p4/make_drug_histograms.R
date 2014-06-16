#!/usr/bin/env Rscript
library(ggplot2)

# File names and variables
dir <- "/Users/christopherwilliams/Desktop/stanford/Fall\ 2013/CS274/projects/p4/"
sunetID             <- "ccwillia"
f.pairwise.tanimoto <- paste0(dir, "all_pairwise_tanimoto.csv")
f.all               <- paste0(dir, "tosubmit/all_tanimoto.png")
f.shared            <- paste0(dir, "tosubmit/shared_tanimoto.png")
f.notshared         <- paste0(dir, "tosubmit/notshared_tanimoto.png")
f.colored           <- paste0(dir, "colored_tanimoto.png")

binwidth            <- 0.025
hist.limits         <- xlim(c(0, 1))


# Read data into a data.frame
df.tanimoto <- read.table(f.pairwise.tanimoto,
                          as.is       = TRUE,
                          sep         = ",", 
                          col.names   = c("drug1", "drug2", 
                                          "tanimoto.score", "share.target"),
                          colClasses = c("character", "character",
                                          "numeric", "factor"))

# Make plots
p.all <- ggplot(df.tanimoto, aes(x=tanimoto.score)) +
    geom_histogram(binwidth=binwidth, color="gray50") +
    geom_hline() + geom_vline() + 
    xlab("Pair-wise Tanimoto scores") + 
    ylab("# drug pairs") +
    ggtitle(paste(sunetID, "All")) + 
    theme_bw() + hist.limits

p.shared <- ggplot(subset(df.tanimoto, share.target == 1), aes(x=tanimoto.score)) +
    geom_histogram(binwidth=binwidth, color="gray50") +
    geom_hline() + geom_vline() + 
    xlab("Pair-wise Tanimoto scores") + 
    ylab("# drug pairs") +
    ggtitle(paste(sunetID, "Shared")) + 
    theme_bw() + hist.limits

p.notshared <- ggplot(subset(df.tanimoto, share.target == 0), aes(x=tanimoto.score)) +
    geom_histogram(binwidth=binwidth, color="gray50") +
    geom_hline() + geom_vline() + 
    xlab("Pair-wise Tanimoto scores") + 
    ylab("# drug pairs") +
    ggtitle(paste(sunetID, "Not Shared")) + 
    theme_bw() + hist.limits

p.colored <- ggplot(df.tanimoto, aes(x=tanimoto.score, fill=share.target)) +
    geom_histogram(binwidth=binwidth, color="gray50") +
    geom_hline() + geom_vline() + 
    xlab("Pair-wise Tanimoto scores") + 
    ylab("# drug pairs") +
    ggtitle(paste(sunetID, "All")) + 
    facet_wrap(~share.target, ncol=1, scales="free_y") +
    scale_fill_manual(values=c("orangered","dodgerblue")) +
    theme_bw() + hist.limits


# Save plots
png(f.all)
p.all
dev.off()

png(f.shared)
p.shared
dev.off()

png(f.notshared)
p.notshared
dev.off()

png(f.colored)
p.colored
dev.off()
