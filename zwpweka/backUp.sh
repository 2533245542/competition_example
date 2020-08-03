#scp -r ../zwpweka wzhou87@app14:backup/zwpwekaBackup
rsync -av -e ssh --exclude='*.zip' --exclude='*.log' --exclude='*.git*' ../zwpweka wzhou87@app14:backup/zwpwekaBackup
