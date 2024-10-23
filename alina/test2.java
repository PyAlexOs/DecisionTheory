package com.example.kursovaya;

import android.content.Context;
import android.content.Intent;
import android.media.session.PlaybackState;
import android.os.Build;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.AnimatorRes;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.recyclerview.widget.RecyclerView;

import java.text.BreakIterator;
import java.util.ArrayList;

//этот пользовательский адаптер расширяет райсикл вью
public class Adapter extends RecyclerView.Adapter<Adapter.holder> {

    private Context context;
    private ArrayList tx, art, name, about, count, price;
    int pos;

    Adapter(Context context, ArrayList art, ArrayList tx, ArrayList name, ArrayList about, ArrayList count, ArrayList price){
        this.context = context;
        this.tx = tx;
        this.name = name;
        this.art = art;
        this.about = about;
        this.count = count;
        this.price = price;
    }

    @NonNull
    @Override
    public holder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        LayoutInflater inflater = LayoutInflater.from(context);
        View view = inflater.inflate(R.layout.str, parent, false);
        return new holder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull final holder holder, final int pos) {

        holder.txt.setText(String.valueOf(tx.get(pos)));
        holder.name_txt.setText(String.valueOf(name.get(pos)));
        holder.art_txt.setText(String.valueOf(art.get(pos)));
        holder.about_txt.setText(String.valueOf(about.get(pos)));
        holder.count_txt.setText(String.valueOf(count.get(pos)));
        holder.price_txt.setText(String.valueOf(price.get(pos)));

        holder.main_lay.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(context, InsideActivity.class);
                context.startActivity(intent);
            }
        });
    }

    @Override
    public int getItemCount() {
        return tx.size();
    }

    public static class holder extends RecyclerView.ViewHolder{

        TextView txt, art_txt, name_txt, about_txt, count_txt, price_txt;
        LinearLayout main_lay;
        public holder(@NonNull View itemView) {
            super(itemView);
            txt = itemView.findViewById(R.id.txt);
            name_txt = itemView.findViewById(R.id.name_txt);
            art_txt = itemView.findViewById(R.id.art_txt);
            about_txt = itemView.findViewById(R.id.about_txt);
            count_txt = itemView.findViewById(R.id.count_txt);
            price_txt = itemView.findViewById(R.id.price_txt);

            main_lay = itemView.findViewById(R.id.main_lay);
        }
    }

}