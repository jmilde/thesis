for i in {1..20}
do
    if [ $i -eq 20 ]
    then
        python3 -m src.train_introvae train_0 last
    else
        python3 -m src.train_introvae train_0 other
    fi
done
