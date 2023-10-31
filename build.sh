cd python
rm -rf build
python setup.py bdist_wheel
pip uninstall pyanns -y
cd dist
ls | xargs pip install
