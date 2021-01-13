#!/bin/bash
# Sourced from: http://ryan.himmelwright.net/post/scripting-tmux-workspaces/

# Session Name
session="Computing"
SESSIONEXISTS=$(tmux list-sessions | grep $SESSION)

# Only create tmux session if it doesn't already exist
if [ "$SESSIONEXISTS" = "" ]
then
    # Start New Session with our name
    tmux new-session -d -s $session
fi

# Attach Session, on the Main window
tmux attach-session -t $SESSION:0