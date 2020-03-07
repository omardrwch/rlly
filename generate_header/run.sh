# create rlly.hpp
OPTION=$1
rm -rf generate_header/all_files
python generate_header/preprocess_rlly.py $OPTION
python generate_header/acme.py generate_header/all_files/rlly.hpp
mv generate_header/all_files/output/rlly.hpp rlly.hpp
rm -rf generate_header/all_files
