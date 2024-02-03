if [ "$1" = "train" ]; then
  python "$1".py "$2" "$3"
else
  python "$1".py "$2" "$3" "$4"
fi
