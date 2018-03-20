#!/usr/bin/env bash

# Information can be found here :
# https://www.tecmint.com/sshfs-mount-remote-linux-filesystem-directory-using-ssh/
#

sudo sshfs -o allow_other $1@baobab.unige.ch:/sst1m/ /sst1m/