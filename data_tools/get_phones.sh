for dir_name in davis_corpus/*
do
	phonfreq +fS +re $dir_name/*
done