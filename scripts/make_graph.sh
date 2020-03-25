#!/usr/bin/env bash

func(){
  python utils/make_graph.py -i $1
}

func 1&
func 2&
func 3&
func 4&
func 5&
func 6&
func 7&
func 8&
func 9&
func 10&
func 11&
func 12&