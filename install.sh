# installing python 3.5 from source on Ubuntu 18.04: https://vlearningit.wordpress.com/install-python3-5-from-the-source-in-ubuntu-18-04/
PROJECT=strips-seg
BASE_DIR=$HOME
PROJECT_DIR=`pwd`
ENV_DIR=$BASE_DIR/envs/$PROJECT # directory for the virtual environemnt
PYTHON_VERSION=6
ORANGE='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

echo -e "${ORANGE}1) Preparing environment${NC}"
mkdir -p $ENV_DIR
sudo apt update
sudo apt install python3.$PYTHON_VERSION-dev python3.$PYTHON_VERSION-tk python3-pip -y
sudo pip3 install -U virtualenv
virtualenv -p python3.$PYTHON_VERSION $BASE_DIR/envs/$PROJECT

echo -e "${ORANGE} 2) Installing Python requirements${NC}"
. $ENV_DIR/bin/activate
cd $PROJECT_DIR
pip install -r requirements.txt