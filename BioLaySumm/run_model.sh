if [ "$1" = "train" ]; then
  python run.py "$1" "$2" "$3"
else
  python run.py "$1" "$2" "$3" "$4"
fi
