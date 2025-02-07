#!/bin/bash

# Get the list of clusters and iterate over each one
for cluster in $(kubectl --context interlinked get clusters --no-headers=true | awk '{print $1}'); do
  echo "Syncing cluster: $cluster"
  i9d sync "$cluster"
done