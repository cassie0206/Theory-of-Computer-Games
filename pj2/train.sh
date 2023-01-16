./threes --total=0 --slide="save=weights.bin" # generate a clean network
for i in {1..100}; do
	./threes --total=100000 --block=1000 --limit=1000 --slide="load=weights.bin save=weights.bin alpha=0.003125" | tee -a train.log
	./threes --total=1000 --slide="load=weights.bin alpha=0" --save="stats.txt"
	tar zcvf weights.$(date +%Y%m%d-%H%M%S).tar.gz weights.bin train.log stats.txt
done